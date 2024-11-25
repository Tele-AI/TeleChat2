"""Evaluate models."""
import os
import argparse
from collections import defaultdict

from typing import List
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
https://huggingface.co/datasets/haonan-li/cmmlu/resolve/main/cmmlu_v1_0_1.zip
"""

task2desc = {
    "agronomy": "农学",
    "anatomy": "解剖学",
    "ancient_chinese": "古汉语",
    "arts": "艺术学",
    "astronomy": "天文学",
    "business_ethics": "商业伦理",
    "chinese_civil_service_exam": "中国公务员考试",
    "chinese_driving_rule": "中国驾驶规则",
    "chinese_food_culture": "中国饮食文化",
    "chinese_foreign_policy": "中国外交政策",
    "chinese_history": "中国历史",
    "chinese_literature": "中国文学",
    "chinese_teacher_qualification": "中国教师资格",
    "clinical_knowledge": "临床知识",
    "college_actuarial_science": "大学精算学",
    "college_education": "大学教育学",
    "college_engineering_hydrology": "大学工程水文学",
    "college_law": "大学法律",
    "college_mathematics": "大学数学",
    "college_medical_statistics": "大学医学统计",
    "college_medicine": "大学医学",
    "computer_science": "计算机科学",
    "computer_security": "计算机安全",
    "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理",
    "economics": "经济学",
    "education": "教育学",
    "electrical_engineering": "电气工程",
    "elementary_chinese": "小学语文",
    "elementary_commonsense": "小学常识",
    "elementary_information_and_technology": "小学信息技术",
    "elementary_mathematics": "初等数学",
    "ethnology": "民族学",
    "food_science": "食品科学",
    "genetics": "遗传学",
    "global_facts": "全球事实",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "high_school_geography": "高中地理",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理学",
    "high_school_politics": "高中政治",
    "human_sexuality": "人类性行为",
    "international_law": "国际法学",
    "journalism": "新闻学",
    "jurisprudence": "法理学",
    "legal_and_moral_basis": "法律与道德基础",
    "logical": "逻辑学",
    "machine_learning": "机器学习",
    "management": "管理学",
    "marketing": "市场营销",
    "marxist_theory": "马克思主义理论",
    "modern_chinese": "现代汉语",
    "nutrition": "营养学",
    "philosophy": "哲学",
    "professional_accounting": "专业会计",
    "professional_law": "专业法学",
    "professional_medicine": "专业医学",
    "professional_psychology": "专业心理学",
    "public_relations": "公共关系",
    "security_study": "安全研究",
    "sociology": "社会学",
    "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药",
    "virology": "病毒学",
    "world_history": "世界历史",
    "world_religions": "世界宗教",
}


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


def format_example(line, subject, include_answer=True):
    """Example format."""
    example = f"以下是关于{task2desc.get(subject, None)}的单项选择题，请直接给出正确答案的选项。\n\n"
    example = example + "题目：" + line["Question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\n答案：" + line["Answer"] + "\n\n"
    else:
        example += "\n答案："
    return example


def generate_few_shot_prompt(k, subject, dev_df):
    """Generate prompt."""
    prompt = ""
    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            subject,
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
            correct_ratio = 100 * existing_df["correctness"].sum() / len(existing_df["correctness"])
            return correct_ratio
    result = []
    score = []

    few_shot_prompt = (
        generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
    )
    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}
    if global_args.debug:
        print(f"few_shot_prompt: {few_shot_prompt}")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, subject_name, include_answer=False)
        full_prompt = few_shot_prompt + question

        # pylint: disable=W0612
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

        if "Answer" in row:
            correct = 1 if pred == row["Answer"] else 0
            score.append(correct)
            if global_args.debug:
                print(f'{question} pred: {pred} ref: {row["Answer"]}')
        result.append(pred)

    if score:
        correct_ratio = 100 * sum(score) / len(score)
        if global_args.debug:
            print(subject_name, correct_ratio)
    else:
        correct_ratio = 0
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

    return correct_ratio


def cal_cmmlu(res):
    print("\n\n\n")
    res = {k.split("-")[-1]: float(v) for k, v in res.items()}
    for k, v in TASK_NAME_MAPPING.items():
        avg_acc = np.mean(list(map(lambda x: res[x], v)))
        print(f"{k} acc: {avg_acc:.2f}")
    avg_all_acc = np.mean(list(res.values()))
    print(f"AVERAGE acc: {avg_all_acc:.2f}")


subcategories = {
    "agronomy": ["other"],
    "anatomy": ["biology"],
    "ancient_chinese": ["linguistics", "china specific"],
    "arts": ["arts"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "chinese_civil_service_exam": ["politics", "china specific"],
    "chinese_driving_rule": ["other", "china specific"],
    "chinese_food_culture": ["culture", "china specific"],
    "chinese_foreign_policy": ["politics", "china specific"],
    "chinese_history": ["history", "china specific"],
    "chinese_literature": ["literature", "china specific"],
    "chinese_teacher_qualification": ["education", "china specific"],
    "college_actuarial_science": ["math"],
    "college_education": ["education"],
    "college_engineering_hydrology": ["engineering"],
    "college_law": ["law"],
    "college_mathematics": ["math"],
    "college_medical_statistics": ["statistics"],
    "clinical_knowledge": ["other"],
    "college_medicine": ["other"],
    "computer_science": ["computer science"],
    "computer_security": ["other"],
    "conceptual_physics": ["physics"],
    "construction_project_management": ["other", "china specific"],
    "economics": ["economics"],
    "education": ["education"],
    "elementary_chinese": ["linguistics", "china specific"],
    "elementary_commonsense": ["other", "china specific"],
    "elementary_information_and_technology": ["other"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "ethnology": ["culture", "china specific"],
    "food_science": ["other"],
    "genetics": ["biology"],
    "global_facts": ["global"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_geography": ["geography"],
    "high_school_mathematics": ["math"],
    "high_school_physics": ["physics"],
    "high_school_politics": ["politics", "china specific"],
    "human_sexuality": ["other"],
    "international_law": ["law"],
    "journalism": ["sociology"],
    "jurisprudence": ["law"],
    "legal_and_moral_basis": ["other"],
    "logical": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "marxist_theory": ["philosophy"],
    "modern_chinese": ["linguistics", "china specific"],
    "nutrition": ["other"],
    "philosophy": ["philosophy"],
    "professional_accounting": ["business"],
    "professional_law": ["law"],
    "professional_medicine": ["other"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_study": ["politics"],
    "sociology": ["culture"],
    "sports_science": ["other"],
    "traditional_chinese_medicine": ["other", "china specific"],
    "virology": ["biology"],
    "world_history": ["history"],
    "world_religions": ["global"],
}

categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
        "statistics",
    ],
    "Humanities": ["history", "philosophy", "law", "arts", "literature", "global"],
    "Social Science": [
        "linguistics",
        "business",
        "politics",
        "culture",
        "economics",
        "geography",
        "psychology",
        "education",
        "sociology",
    ],
    "Other": ["other"],
    "China specific": ["china specific"],
}

TASK_NAME_MAPPING = defaultdict(list)


def init_task_name_mapping():
    global TASK_NAME_MAPPING
    for k, v in categories.items():
        for subject, subcat in subcategories.items():
            for c in subcat:
                if c in v:
                    TASK_NAME_MAPPING[k].append(subject)

choices = ["A", "B", "C", "D"]


def main(args):
    print(args)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=args.device_id)
    pipeline_task, tokenizer = load_models_tokenizer(args)

    test_result = {}
    for subject_name in tqdm(subcategories.keys()):
        dev_file_path = os.path.join(args.eval_data_path, "dev", f"{subject_name}.csv")
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}.csv"
        )
        dev_df = pd.read_csv(dev_file_path)
        test_df = pd.read_csv(test_file_path)

        score = eval_subject(
            pipeline_task,
            tokenizer,
            subject_name,
            dev_df=dev_df,
            test_df=test_df,
            k=5,
            few_shot=True,
            save_result_dir=f"outs/cmmlu_eval_result_lite",
        )
        test_result[subject_name] = score
    cal_cmmlu(test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument("-c", "--checkpoint-path", type=str, help="Checkpoint path", default="")
    parser.add_argument("-t", "--token_path", type=str, help="Tokenizer.model path", default="")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")

    # pylint: disable=W0105
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, required=True, help="Path to eval data")
    parser.add_argument("--device_id", type=int, default=0, help="Device id")
    group.add_argument("--max-seq-len", type=int, default=2048, help="Size of the output generated text.")
    group.add_argument("--debug", action="store_true", default=False, help="Print infos.")
    group.add_argument("--config", type=str, required=False, help="Path to config")
    parser.add_argument('--full_model_path', default=None, type=str, help="load mindir full checkpoint")
    parser.add_argument('--inc_model_path', default=None, type=str, help="load mindir inc checkpoint")
    group.add_argument("--config_path", type=str, required=False, help="Path to GE config")

    global_args = parser.parse_args()
    set_seed(global_args.seed)

    init_task_name_mapping()

    main(global_args)
