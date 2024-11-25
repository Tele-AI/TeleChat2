# Copyright 2023 Huawei Technologies Co., Ltd
#
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
"""Image-to-text Retrieval Trainer Utils."""
import numpy as np
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset

from mindformers.tools.logger import logger
from mindformers.generation.utils import topk


def append_text_outputs(eval_network, text_input, text_feats, text_ids):
    """append_text_outputs

    Args:
        text_input: text_input, can be:
                    1) 2-dimension: 1 image 1 caption
                    2) 3-dimension: 1 image multiple captions
        text_feats: output to append
        text_ids: output to append
    """
    if text_input.ndim == 3:
        for input_ids in text_input:
            _, text_feat = eval_network.text_forwarder(input_ids)
            text_feats.append(text_feat.asnumpy())
            text_ids.append(input_ids.asnumpy())
    else:
        _, text_feat = eval_network.text_forwarder(text_input)
        text_feats.append(text_feat.asnumpy())
        text_ids.append(text_input.asnumpy())


def append_image_outputs(eval_network, image_input, image_feats, vit_outputs):
    """append_image_outputs

    Args:
        image_input: image_input, can be:
                    1) 4-dimension: 1 caption 1 image
                    2) 5-dimension: 1 caption multiple images
        image_feats: output to append
        vit_outputs: output to append
    """
    if image_input.ndim == 5:
        for image in image_input:
            image_feat, vit_output = eval_network.image_forwarder(image)
            image_feats.append(image_feat.asnumpy())
            vit_outputs.append(vit_output.asnumpy())
    else:
        image_feat, vit_output = eval_network.image_forwarder(image_input)
        image_feats.append(image_feat.asnumpy())
        vit_outputs.append(vit_output.asnumpy())


def prepare_inputs_for_itm_eval(eval_network, dataloader):
    """
    prepare inputs of BLIP-2 for image_to_text_reatrieval task.

    Args:
        dataloader (GeneratorDataset): image-caption pair dataloader

    Returns:
        image_feats, text_feats, vit_outputs, text_ids, text_atts
    """
    image_feats = []
    text_feats = []
    vit_outputs = []
    text_ids = []
    for image_input, text_input in dataloader:
        append_text_outputs(eval_network, text_input, text_feats, text_ids)
        append_image_outputs(eval_network, image_input, image_feats, vit_outputs)

    image_feats = np.concatenate(image_feats, axis=0)
    text_feats = np.concatenate(text_feats, axis=0)
    vit_outputs = np.concatenate(vit_outputs, axis=0)
    text_ids = np.concatenate(text_ids, axis=0)

    return Tensor.from_numpy(image_feats), \
           Tensor.from_numpy(text_feats),  \
           Tensor.from_numpy(vit_outputs), \
           Tensor.from_numpy(text_ids)


def extract_image_text_mapping(eval_dataloader, score_i2t, score_t2i):
    """extract_image_text_mapping from eval_dataloader.

    Args:
        eval_dataloader: evaluation dataloader
        score_i2t, score_t2i: two score matrix (I2T, T2I)

    Returns:
        img2txt, txt2img: ground truth image-text mapping
    """
    dataloader = eval_dataloader
    while not isinstance(dataloader, GeneratorDataset) and hasattr(dataloader, "children") \
        and dataloader.children is not None:
        dataloader = dataloader.children[0]
    if isinstance(dataloader, GeneratorDataset):
        dataset = dataloader.source
        if hasattr(dataset, 'img2txt') and hasattr(dataset, 'txt2img'):
            logger.info("loading img2txt and txt2img from eval_dataset...")
            return dataset.img2txt, dataset.txt2img
    logger.warning("expect the eval dataset to be generate from " \
                   "MultiImgCapDataLoader.img2txt/txt2img, but not succeeded. " \
                   "will generate image-text mapping with accumulate indexes by default.")
    if (score_i2t.shape[0], score_i2t.shape[1]) != (score_t2i.shape[1], score_t2i.shape[0]):
        raise ValueError(f"(score_i2t.shape[0], score_i2t.shape[1]) "
                         f"should equal to (score_t2i.shape[1], score_t2i.shape[0])")
    image_num = score_i2t.shape[0]
    text_num = score_t2i.shape[0]
    if image_num == text_num:
        inds = [i for i in range(image_num)]
        return inds, inds
    bigger = image_num if image_num > text_num else text_num
    smaller = text_num if bigger == image_num else image_num
    factor = bigger // smaller
    smaller_inds = [[min(i + k * factor, bigger) for i in range(factor)] for k in range(smaller)]
    bigger_inds = []
    for i in range(smaller - 1):
        bigger_inds += [i] * factor
    last_num = bigger - (smaller - 1) * factor
    bigger_inds += [smaller - 1] * last_num
    if bigger == image_num:
        return bigger_inds, smaller_inds
    return smaller_inds, bigger_inds


def compute_extra_itm(model, vit_outputs, text_ids, i, k_test, topk_idx, i2t=True):
    """ compute extra_itm score, for those model have
        its own itm computing method, like blip-2.

    Args:
        extra_args (tuple): required args for this method.
        i (int): index in the sim_matrix
        k_test (int): k_test number
        topk_idx (Tensor): topk_idx
        i2t (bool, optional): is image-to-text or text-to-image matching.

    Returns:
        _type_: extra itm score
    """
    if i2t:
        image_inputs = np.tile(vit_outputs[i], (k_test, 1, 1))
        score = model.itm_computer(
            image_inputs=Tensor.from_numpy(image_inputs),
            text_ids=Tensor.from_numpy(text_ids[topk_idx])
        )
    else:
        image_inputs = vit_outputs[topk_idx]
        score = model.itm_computer(
            image_inputs=Tensor.from_numpy(image_inputs),
            text_ids=Tensor.from_numpy(np.tile(text_ids[i], (k_test, 1)))
        )

    return score.asnumpy()


def compute_itm_scores(network,
                       sims_matrix,
                       vit_outputs=None,
                       text_ids=None,
                       k_test=128,
                       add_extra_itm_score=False,
                       log_level=50):
    """
    compute image-text matching scores, in matrix format.

    Args:
        network (PreTrainedModel): network for evaluate ITM
        eval_inputs (tuple): inputs for evaluate itm scores.
        k_test (int, optional): k_test num, Defaults to 128.
        add_extra_itm_score (bool, optional): whether to add extra scores (model decides), Defaults to False.
        log_level (int, optional): the log level to show progress in itm score computing.

    Returns:
        score_matrix_i2t, score_matrix_t2i: two score matrix (I2T, T2I)
    """
    logger.info("Start compute_itm_scores ...")
    # get distributed info
    rank = network.rank
    group_size = network.group_size

    # data parallel computing
    total = sims_matrix.shape[0]
    step = total // group_size
    start = rank * step
    if rank == group_size - 1:
        step = total - start
    end = start + step
    sims_matrix_sub = sims_matrix[start: end]
    logger.info("rank_%d I2T - start: %d | end: %d | total: %d", rank, start, end, total)

    # mask output
    score_matrix_i2t = np.zeros(
        (sims_matrix.shape[0], sims_matrix.shape[1]),
        dtype=np.float32
    )
    score_matrix_i2t[start: end] = -100.0

    # i2t
    for i, sims in enumerate(sims_matrix_sub):
        if i % log_level == 0:
            logger.info("evaluated: %d/%d - sims.shape: (%d,)", i, step, sims.shape[0])
        topk_sim, topk_idx = topk(sims.T, k_test)
        topk_sim = topk_sim.T
        topk_idx = topk_idx.T
        score_matrix_i2t[i + start, topk_idx] = topk_sim # .astype(mstype.float32)
        if add_extra_itm_score:
            score = compute_extra_itm(network, vit_outputs, text_ids, i + start, k_test, topk_idx, i2t=True)
            score_matrix_i2t[i + start, topk_idx] += score

    # switch from i2t -> t2i
    sims_matrix = sims_matrix.T

    # data parallel computing
    total = sims_matrix.shape[0]
    step = total // group_size
    start = rank * step
    if rank == group_size - 1:
        step = total - start
    end = start + step
    sims_matrix_sub = sims_matrix[start: end]
    logger.info("rank_%d T2I - start: %d | end: %d | total: %d", rank, start, end, total)

    # mask output
    score_matrix_t2i = np.zeros(
        (sims_matrix.shape[0], sims_matrix.shape[1]),
        dtype=np.float32
    )
    score_matrix_t2i[start: end] = -100.0

    # t2i
    for i, sims in enumerate(sims_matrix_sub):
        if i % log_level == 0:
            logger.info("evaluated: %d/%d - sims.shape: (%d,)", i, step, sims.shape[0])
        topk_sim, topk_idx = topk(sims.T, k_test)
        topk_sim = topk_sim.T
        topk_idx = topk_idx.T
        score_matrix_t2i[i + start, topk_idx] = topk_sim
        if add_extra_itm_score:
            score = compute_extra_itm(network, vit_outputs, text_ids, i + start, k_test, topk_idx, i2t=False)
            score_matrix_t2i[i + start, topk_idx] += score

    # score reducer for DP
    score_matrix_i2t = Tensor.from_numpy(score_matrix_i2t)
    score_matrix_t2i = Tensor.from_numpy(score_matrix_t2i)
    score_matrix_i2t, score_matrix_t2i = network.score_reducer(score_matrix_i2t, score_matrix_t2i)
    return score_matrix_i2t.asnumpy(), score_matrix_t2i.asnumpy()


def report_metrics(scores_i2t, scores_t2i, img2txt, txt2img):
    """
    report metrics for image-text matching

    Args:
        scores_i2t: image-to-text similarity score matrix
        scores_t2i: text-to-image similarity score matrix
        img2txt: image-to-text ground truth mapping
        txt2img: text-to-image ground truth mapping

    Returns:
        eval_result: A dictionary containing r1, r5, r10 scores
    """
    def get_lowest_from_ranks(ranks, ground_truth):
        if isinstance(ground_truth, int):
            return np.where(ranks == ground_truth)[0][0]
        if not isinstance(ground_truth, list):
            raise ValueError("img2txt or txt2img should be list[int] or list[list[int]]!")
        rank = 1e20
        for i in ground_truth:
            tmp = np.where(ranks == i)[0][0]
            if tmp < rank:
                rank = tmp
        return rank
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        ranks[index] = get_lowest_from_ranks(inds, img2txt[index])

    # 计算度量, 100 是百分比基数,
    # ranks为每张图ground-truth对应的
    # text在similarity score中排位的最小值。
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = get_lowest_from_ranks(inds, txt2img[index])

    # 此段逻辑同上注释，以text为基准。
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    agg_metrics = (tr1 + tr5 + tr10) / 3

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
        "agg_metrics": agg_metrics,
    }
    return eval_result
