# Copyright 2024 VLMEvalKit Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Files used for model evaluation."""
import ssl

# pylint: disable=W0401
from vlmeval.smp import *
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.config import supported_VLM

from toolkit.benchmarks.multimodal_models import get_model

# pylint: disable=protected-access
ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    """Describe evaluation parameters."""
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True, help='setup evaluate data')
    parser.add_argument('--model', type=str, nargs='+', required=True, help='setup models')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='.', help='select the output directory')
    # Logging Utils
    parser.add_argument('--verbose', action='store_true', help='output log')
    # model Path
    parser.add_argument('--model-path', type=str, nargs='+', required=True, help='setup the model path')
    # config Path
    parser.add_argument('--config-path', type=str, nargs='+', required=True, help='setup the config path')
    args = parser.parse_args()
    return args


def main():
    logger = get_logger('RUN')

    args = parse_args()
    if not args.data:
        raise ValueError("--data should be a list of data files")
    if not args.model_path:
        raise ValueError("--model-path should be a list of model paths")
    if not args.config_path:
        raise ValueError("--config-path should be a list of config paths")

    for _, model_name in enumerate(args.model):
        model = None
        pred_root = osp.join(args.work_dir, model_name)
        os.makedirs(pred_root, exist_ok=True)

        supported_VLM.update(get_model(args))

        for _, dataset_name in enumerate(args.data):
            dataset_kwargs = {}
            # If distributed, first build the dataset on the main process for doing preparation works
            dataset = build_dataset(dataset_name, **dataset_kwargs)
            if dataset is None:
                logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                continue

            result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'
            if model is None:
                model = model_name  # which is only a name
            # Perform the Inference
            model = infer_data_job(
                model,
                work_dir=pred_root,
                model_name=model_name,
                dataset=dataset,
                verbose=args.verbose)

            # Set the judge kwargs first before evaluation or dumping
            judge_kwargs = {
                'verbose': args.verbose,
            }

            data_list = ['MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                         'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11']
            if dataset_name in data_list:
                if not MMBenchOfficialServer(dataset_name):
                    logger.error(
                        f'Can not evaluate {dataset_name} on non-official servers, '
                        'will skip the evaluation. '
                    )
                    continue

            eval_results = dataset.evaluate(result_file, **judge_kwargs)
            if eval_results is not None:
                if not isinstance(eval_results, dict) and not isinstance(eval_results, pd.DataFrame):
                    raise TypeError("eval_results must be either a dict or a pandas DataFrame")
                logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                logger.info('Evaluation Results:')
            if isinstance(eval_results, dict):
                logger.info('\n' + json.dumps(eval_results, indent=4))
            elif isinstance(eval_results, pd.DataFrame):
                if len(eval_results) < len(eval_results.columns):
                    eval_results = eval_results.T
                logger.info('\n' + tabulate(eval_results))


if __name__ == "__main__":
    load_env()
    main()
