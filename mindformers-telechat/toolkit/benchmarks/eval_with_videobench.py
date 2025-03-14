# Copyright 2024 VideoBench.
"""Evaluation of VideoBench for MindFormers multimodal models."""
import argparse
import os
import json
import numpy as np

from mindformers import logger
from mindformers.tools.register.config import MindFormerConfig
from toolkit.benchmarks.vlmevalkit_models.multimodal_models import init_model
from toolkit.benchmarks.vlmevalkit_models.support_models import SUPPORT_MODEL_LIST


def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help="setup the model path",)
    parser.add_argument('--config_path',
                        type=str,
                        required=True,
                        help="setup the config path",)
    parser.add_argument('--dataset_name',
                        type=str,
                        default=None,
                        help="The type of LLM")
    parser.add_argument('--Eval_QA_root',
                        type=str,
                        default='/usr/local/Ascend/Video-Bench/',
                        help="folder containing QA JSON files",)
    parser.add_argument('--Eval_Video_root',
                        type=str,
                        help="folder containing video data",)
    parser.add_argument('--chat_conversation_output_folder',
                        type=str,
                        default='./Chat_results',
                        help="")
    return parser.parse_args()


def create_model(model_path, config_path):
    """Init model."""
    if os.path.exists(config_path) and config_path.endswith('.yaml'):
        config = MindFormerConfig(config_path)
        if config.trainer.model_name not in SUPPORT_MODEL_LIST.get("video"):
            raise ValueError(f"model {config.trainer.model_name} is not support.")
        ms_model = init_model(model_path, config_path)
        return ms_model
    raise ValueError(
        f"the config_path should be a valid yaml file and exist, but got `{config_path}`, please check it.")


if __name__ == '__main__':
    args = parse_arguments()

    eval_qa_root = args.Eval_QA_root
    eval_video_root = args.Eval_Video_root
    dataset_qajson = {
        "Ucfcrime": f"{eval_qa_root}/Eval_QA/Ucfcrime_QA_new.json",
        "Youcook2": f"{eval_qa_root}/Eval_QA/Youcook2_QA_new.json",
        "TVQA": f"{eval_qa_root}/Eval_QA/TVQA_QA_new.json",
        "MSVD": f"{eval_qa_root}/Eval_QA/MSVD_QA_new.json",
        "MSRVTT": f"{eval_qa_root}/Eval_QA/MSRVTT_QA_new.json",
        "Driving-decision-making": f"{eval_qa_root}/Eval_QA/Driving-decision-making_QA_new.json",
        "NBA": f"{eval_qa_root}/Eval_QA/NBA_QA_new.json",
        "SQA3D": f"{eval_qa_root}/Eval_QA/SQA3D_QA_new.json",
        "Driving-exam": f"{eval_qa_root}/Eval_QA/Driving-exam_QA_new.json",
        "MV": f"{eval_qa_root}/Eval_QA/MV_QA_new.json",
        "MOT": f"{eval_qa_root}/Eval_QA/MOT_QA_new.json",
        "ActivityNet": f"{eval_qa_root}/Eval_QA/ActivityNet_QA_new.json",
        "TGIF": f"{eval_qa_root}/Eval_QA/TGIF_QA_new.json",
    }

    if not args.model_path:
        raise ValueError("--model-path should be a str of model path")
    if not args.config_path:
        raise ValueError("--config-path should be a str of config path")
    if args.dataset_name is None:
        dataset_name_list = list(dataset_qajson.keys())
    else:
        dataset_name_list = [args.dataset_name]
        logger.info(f'Specifically run {args.dataset_name}')
    logger.info(dataset_name_list)

    os.makedirs(args.chat_conversation_output_folder, exist_ok=True)

    model_output = create_model(args.model_path, args.config_path)
    model = model_output.model
    processor = model_output.processor
    batch_size = model_output.batch_size
    tokenizer = model_output.tokenizer

    for dataset_name in dataset_name_list:
        qa_json = dataset_qajson.get(dataset_name)
        logger.info(f'Dataset name:{dataset_name}, qa_json:{qa_json}!')
        with open(qa_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        eval_dict = {}
        for idx, (q_id, item) in enumerate(data.items()):
            try:
                video_id = item['video_id']
                question = item['question']
                if len(item['choices']) == 6:
                    question += (f"Choices: A.{item['choices']['A']} B.{item['choices']['B']} "
                                 f"C.{item['choices']['C']} D.{item['choices']['D']} E.{item['choices']['E']} "
                                 f"F.{item['choices']['F']} \n Among the six options A, B, C, D, E, F above, "
                                 f"the one closest to the correct answer is:")
                    candidates = ['A', 'B', 'C', 'D', 'E', 'F']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}",
                                       f"C.{item['choices']['C']}", f"D.{item['choices']['D']}",
                                       f"E.{item['choices']['E']}", f"F.{item['choices']['F']}"]
                elif len(item['choices']) == 5:
                    question += (f" A.{item['choices']['A']} B.{item['choices']['B']} "
                                 f"C.{item['choices']['C']} D.{item['choices']['D']} "
                                 f"E.{item['choices']['E']} \n Among the five options A, B, C, D, E above, "
                                 f"the one closest to the correct answer is: ")
                    candidates = ['A', 'B', 'C', 'D', 'E']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}",
                                       f"C.{item['choices']['C']}", f"D.{item['choices']['D']}",
                                       f"E.{item['choices']['E']}"]
                elif len(item['choices']) == 4:
                    question += (f" A.{item['choices']['A']} B.{item['choices']['B']} "
                                 f"C.{item['choices']['C']} D.{item['choices']['D']} \n "
                                 f"Among the four options A, B, C, D above, the one closest to the correct answer is:")
                    candidates = ['A', 'B', 'C', 'D']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}",
                                       f"C.{item['choices']['C']}", f"D.{item['choices']['D']}"]
                elif len(item['choices']) == 3:
                    question += (f" A.{item['choices']['A']} B.{item['choices']['B']} "
                                 f"C.{item['choices']['C']} \n Among the three options A, B, C above, "
                                 f"the one closest to the correct answer is: ")
                    candidates = ['A', 'B', 'C']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}",
                                       f"C.{item['choices']['C']}"]
                elif len(item['choices']) == 2:
                    question += (f" A.{item['choices']['A']} B.{item['choices']['B']} \n "
                                 f"Among the two options A, B above, the one closest to the correct answer is: ")
                    candidates = ['A', 'B']
                    candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}"]
                vid_rela_path = item['vid_path']
                vid_path = os.path.join(eval_video_root, vid_rela_path)

                message = [{"video": vid_path}, {"text": question}] * batch_size
                input_data = processor(message)
                res = model.generate(**input_data)
                input_id_length = np.max(np.argwhere(input_data.get("input_ids")[0] != tokenizer.pad_token_id)) + 1
                output = tokenizer.decode(res[0][input_id_length:], skip_special_tokens=True)
                eval_dict[q_id] = {
                    'video_id': video_id,
                    'question': question,
                    'output_sequence': output
                }
                logger.info(f'q_id:{q_id}, output:{output}!\n')
            except Exception as e:
                raise Exception from e
        # eval results
        eval_dataset_json = f'{args.chat_conversation_output_folder}/{dataset_name}_eval.json'
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(eval_dataset_json, flags_, 0o750), "w", encoding='utf-8') as f:
            json.dump(eval_dict, f, indent=2)
