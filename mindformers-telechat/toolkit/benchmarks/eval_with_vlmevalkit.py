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
from vlmeval.inference_video import infer_data_job_video
from vlmeval.dataset.image_vqa import ImageVQADataset

from toolkit.benchmarks.vlmevalkit_models.support_models import get_model
from toolkit.benchmarks.vlmevalkit_models.support_models import SUPPORT_MODEL_LIST

# pylint: disable=protected-access
ssl._create_default_https_context = ssl._create_unverified_context
cache_path = get_cache_path('OpenGVLab/MVBench', branch='main')


def parse_args():
    """Describe evaluation parameters."""
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True, help='setup evaluate data')
    parser.add_argument('--model', type=str, required=True, help='setup models')
    # Args that only apply to Video Dataset
    parser.add_argument('--nframe', type=int, default=8)
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--use-subtitle', action='store_true')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Logging Utils
    parser.add_argument('--verbose', action='store_true', help='output log')
    # model Path
    parser.add_argument('--model-path', type=str, required=True, help='setup the model path')
    # config Path
    parser.add_argument('--config-path', type=str, required=True, help='setup the config path')
    args = parser.parse_args()
    if not args.data:
        raise ValueError("--data should be a list of data files")
    if not args.model_path:
        raise ValueError("--model-path should be a str of model path")
    if not args.config_path:
        raise ValueError("--config-path should be a str of config path")
    return args


def create_dataset(dataset_name, args):
    """Create dataset."""
    dataset_kwargs = {}
    if dataset_name == 'MMBench-Video':
        dataset_kwargs['pack'] = args.pack
    if dataset_name == 'OCRVQA_TEST':
        if 'LMUData' in os.environ:
            lmu_data_dir = os.environ['LMUData']
            file_path = os.path.join(lmu_data_dir, 'OCRVQA_TEST.tsv')
            if os.path.isfile(file_path):
                ImageVQADataset.DATASET_MD5['OCRVQA_TEST'] = None

    dataset = build_dataset(dataset_name, **dataset_kwargs)
    if dataset_name == 'MVBench':
        data_file = osp.join(cache_path, f'{dataset_name}.tsv')
        if os.path.exists(data_file):
            df = pd.read_csv(data_file, sep='\t')

            def check_path_exists(row):
                full_path = os.path.join(cache_path, row['prefix'], row['video'])
                return os.path.exists(full_path)

            df_cleaned = df[df.apply(check_path_exists, axis=1)]
            df_cleaned.to_csv(data_file, sep='\t', index=False)
    return dataset


def dump_evalresult(eval_results, model_name, dataset_name, logger):
    """Dump result of evaluate."""
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


def main():
    logger = get_logger('RUN')

    args = parse_args()

    model_name = args.model
    if model_name not in SUPPORT_MODEL_LIST.values():
        raise ValueError(f"model {model_name} is not support.")
    model = None
    pred_root = osp.join(args.work_dir, model_name)
    os.makedirs(pred_root, exist_ok=True)

    supported_VLM.update(get_model(args))

    for _, dataset_name in enumerate(args.data):
        try:
            dataset = create_dataset(dataset_name, args)
            if dataset is None:
                logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                continue

            result_file = f'{pred_root}/{model_name}_{dataset_name}.xlsx'
            if dataset_name in ['MMBench-Video']:
                packstr = 'pack' if args.pack else 'nopack'
                result_file = f'{pred_root}/{model_name}_{dataset_name}_{args.nframe}frame_{packstr}.xlsx'
            elif dataset.MODALITY == 'VIDEO':
                if args.pack:
                    logger.info(f'{dataset_name} not support Pack Mode, directly change to unpack')
                    args.pack = False
                packstr = 'pack' if args.pack else 'nopack'
                result_file = f'{pred_root}/{model_name}_{dataset_name}_{args.nframe}frame_{packstr}.xlsx'

            if model is None:
                model = model_name

            # Perform the Inference
            if dataset.MODALITY == 'VIDEO':
                model = infer_data_job_video(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    nframe=args.nframe,
                    pack=args.pack,
                    verbose=args.verbose,
                    subtitle=args.use_subtitle)
            else:
                model = infer_data_job(
                    model,
                    work_dir=pred_root,
                    model_name=model_name,
                    dataset=dataset,
                    verbose=args.verbose)

            # Set the judge kwargs first before evaluation or dumping
            judge_kwargs = {'verbose': args.verbose}
            if listinstr(['MMVet', 'MathVista', 'LLaVABench', 'MMBench-Video', 'MathVision'],
                         dataset_name):
                judge_kwargs['model'] = 'gpt-4-turbo'
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
            dump_evalresult(eval_results, model_name, dataset_name, logger)
        # pylint: disable=broad-except
        except Exception as e:
            logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                             'skipping this combination.')
            continue


if __name__ == '__main__':
    load_env()
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    main()
