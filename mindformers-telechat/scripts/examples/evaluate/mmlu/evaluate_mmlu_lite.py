"""Evaluate models."""
import os
from typing import List
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

import mindspore as ms
from mindspore.common import set_seed
from mindformers import LlamaTokenizer
from mindformers.inference import InferConfig, InferTask
from mindformers.generation.utils import softmax

# pylint: disable=W0105
"""
数据地址
https://people.eecs.berkeley.edu/~hendrycks/data.tar
"""


def load_models_tokenizer(args):
    """Load models tokenizer."""
    tokenizer = LlamaTokenizer(args.token_path)

    lite_config = InferConfig(
        prefill_model_path=args.full_model_path,
        increment_model_path=args.inc_model_path,
        model_type="mindir",
        model_name="llama",
        ge_config_path=args.config_path,
        device_id=args.device_id,
        infer_seq_length=4096,
    )

    pipeline_task = InferTask.get_infer_task("text_generation", lite_config, tokenizer=tokenizer)

    return pipeline_task, tokenizer


def format_example(line, include_answer=True):
    """Example format."""
    example = "Question: " + line["question"] + "\nChoices:\n"
    for choice in choices:
        example += f'{choice}. {line[f"{choice}"]}\n'

    if include_answer:
        example += "Answer: " + line["answer"] + "\n\n"
    else:
        example += "Answer:"
    return example


def generate_few_shot_prompt(k, subject, dev_df):
    """Generate prompt."""

    def format_subject(subject):
        parts = subject.split("_")
        s = ""
        for entry in parts:
            s += " " + entry
        return s.strip()

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )

    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt


def get_logits(tokenizer, pipeline_task, inputs: List[str]):
    """Process a batch of text input inputs and return the model's output logits."""
    input_ids = tokenizer(
        inputs, padding="max_length", max_length=4096, truncation=True, truncate_direction="LEFT")["input_ids"]

    valid_length = []
    valid_length.append(np.max(np.argwhere(np.array(input_ids[0]) != tokenizer.pad_token_id)) + 1)
    valid_length = np.array(valid_length, np.int32)
    current_index = [valid_length[0] - 1]
    current_index = np.array(current_index, np.int32)
    input_ids = np.array(input_ids, np.int32)
    lite_inputs = pipeline_task.get_predict_inputs(pipeline_task.full_model, input_ids, current_index)
    outputs = pipeline_task.full_model.predict(lite_inputs)
    outputs = outputs[0].get_data_to_numpy()
    outputs = np.array([outputs[0][current_index[0], :]])

    logits = softmax(outputs, axis=-1)

    return logits, inputs


# pylint: disable=W0613
# pylint: disable=W0612
def eval_subject(
        pipeline_task,
        tokenizer,
        subject_name,
        test_df,
        k=5,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        **kwargs,
):
    """Evaluate the performance of a test dataset for subject_name"""
    file_path = os.path.join(save_result_dir, f"{subject_name}_result_lite.csv") if save_result_dir else None
    if file_path and os.path.exists(file_path):
        # Read the file, extract the 'correctness' column, and calculate correct_ratio
        existing_df = pd.read_csv(file_path, encoding="utf-8")
        if "correctness" in existing_df:
            return list(existing_df["correctness"])
    result = []
    score = []

    few_shot_prompt = (
        generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
    )
    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
    if global_args.debug:
        print(f"few_shot_prompt: {few_shot_prompt}")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, include_answer=False)
        full_prompt = few_shot_prompt + question

        output, input_info = get_logits(tokenizer, pipeline_task, [full_prompt])
        if output.shape[0] != 1:
            raise ValueError("The output shape is not valid. Expect shape[0] = 1, but got {}".format(output.shape[0]))
        logits = output.flatten()

        softval = softmax(
            np.asarray(
                [
                    logits[tokenizer("A")["input_ids"][-1]],
                    logits[tokenizer("B")["input_ids"][-1]],
                    logits[tokenizer("C")["input_ids"][-1]],
                    logits[tokenizer("D")["input_ids"][-1]],
                ]
            ),
            axis=0,
        )
        if softval.dtype in {np.float16}:
            softval = softval.to(dtype=np.float32)
        probs = softval

        for i, choice in enumerate(choices):
            all_probs.get(f"prob_{choice}").append(probs[i])
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}.get(np.argmax(probs))

        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            if global_args.debug:
                print(f'{question} pred: {pred} ref: {row["answer"]}')
        result.append(pred)

    if save_result_dir:
        test_df["model_output"] = result
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result_lite.csv"),
            encoding="utf-8",
            index=False,
        )

    return score


def cal_mmlu(res):
    """Calculate and print certain metrics stored in a dictionary called res."""
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0
    # pylint: disable=W0612
    hard_cnt = 0
    # pylint: disable=W0612
    hard_acc_sum = 0.0

    for class_, _ in TASK_NAME_MAPPING.items():
        acc_sum_dict[class_] = 0.0
        acc_norm_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    print("\n\n\n", "total cnt:", cnt, "\n")
    for k, _ in TASK_NAME_MAPPING.items():
        if k in cnt_dict:
            print("%s ACC: %.2f " % (k, acc_sum_dict.get(k) / cnt_dict.get(k) * 100))
    if cnt != 0:
        print("AVERAGE ACC:%.2f " % (acc_sum / cnt * 100))


def main(args):
    print(args)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=args.device_id)
    pipeline_task, tokenizer = load_models_tokenizer(args)

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        # val_file_path = os.path.join(args.eval_data_path, 'val', f'{subject_name}_val.csv')
        dev_file_path = os.path.join(
            args.eval_data_path, "dev", f"{subject_name}_dev.csv"
        )
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.csv"
        )
        # val_df = pd.read_csv(val_file_path, names=['question','A','B','C','D','answer'])
        dev_df = pd.read_csv(
            dev_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )

        score = eval_subject(
            pipeline_task,
            tokenizer,
            subject_name,
            test_df,
            dev_df=dev_df,
            k=5,
            few_shot=True,
            save_result_dir=f"outs/mmlu_eval_result_lite",
        )
        dev_result[subject_name] = score
    cal_mmlu(dev_result)


TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c", "--checkpoint-path", type=str, help="Checkpoint path", default="",
    )
    parser.add_argument(
        "-t", "--token_path", type=str, help="Tokenizer.model path", default="",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device_id", type=int, default=0, help="Device id")

    # pylint: disable=W0105
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, help="Path to eval data")
    group.add_argument(
        "--max-seq-len", type=int, default=2048, help="Size of the output generated text.",
    )
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--config", type=str, required=False, help="Path to config"
    )
    parser.add_argument('--full_model_path', default=None, type=str, help="load mindir full checkpoint")
    parser.add_argument('--inc_model_path', default=None, type=str, help="load mindir inc checkpoint")
    group.add_argument("--config_path", type=str, required=False, help="Path to GE config")

    global_args = parser.parse_args()
    set_seed(global_args.seed)

    main(global_args)

# python skywork/evaluate_mmlu_lite.py -d ../skywork_data/mmlu/data --config_path skywork/context.cfg --token_path /home/zxw/skywork-13b_ckpt-new/tokenizer.model --full_model_path output/mindir_full_checkpoint_bs_1/rank_0_graph.mindir --device_id 7
