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
"""Image Classification Pipeline API."""
from typing import Optional, Union
import cv2
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore import Tensor, Model

from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseImageProcessor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.sam import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    box_area,
    nms
)
from .base_pipeline import Pipeline

__all__ = ['SegmentAnythingPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="segment_anything")
class SegmentAnythingPipeline(Pipeline):
    r"""Pipeline for image segment

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        image_processor (Optional[BaseImageProcessor]):
            The image_processor of model, it could be None if the model do not need image_processor.

    Raises:
        TypeError:
            If input model and image_processor's types are not corrected.
        ValueError:
            If the input model is not in support list.
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['segment_anything'].keys()

    def __init__(self, model: Union[PreTrainedModel, Model],
                 image_processor: Optional[BaseImageProcessor] = None,
                 **kwargs):

        if image_processor is None:
            raise ValueError("ImageClassificationFoPipeline"
                             " requires for a image_processor.")

        super().__init__(model, image_processor=image_processor, **kwargs)

        self.reset_config(**kwargs)
        self.stack = ops.Stack(axis=-1)

        self.reset_image()

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.
        """
        preprocess_params = {"seg_image": False}
        forward_params = {"seg_image": False}
        postprocess_params = {"seg_image": False}

        preprocess_list_0 = ["seg_image"]
        for item in preprocess_list_0:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.get(item)
        preprocess_params['reset_config'] = False

        preprocess_list_1 = ["points_per_side", "points_per_batch", "pred_iou_thresh",
                             "stability_score_thresh", "stability_score_offset",
                             "box_nms_thresh", "crop_n_layers", "crop_nms_thresh",
                             "crop_overlap_ratio", "crop_n_points_downscale_factor",
                             "point_grids", "min_mask_region_area", "output_mode"]
        for item in preprocess_list_1:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.get(item)
                preprocess_params['reset_config'] = True

        forward_list = ["seg_image", "multimask_output", "return_logits"]
        for item in forward_list:
            if item in pipeline_parameters:
                forward_params[item] = pipeline_parameters.get(item)

        post_list = ["seg_image"]
        for item in post_list:
            if item in pipeline_parameters:
                postprocess_params[item] = pipeline_parameters.get(item)

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs,
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (dict):
                The image to be classified.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Return:
            Processed image.
        """
        model_inputs = {}
        seg_image = preprocess_params["seg_image"]

        if not seg_image:
            image = inputs.get("image", None)
            if image is not None:
                input_image, original_size, input_size = self.preprocess_image(image)
                model_inputs["image"] = input_image
                model_inputs["features"] = None
                model_inputs["original_size"] = original_size
                model_inputs["input_size"] = input_size
            else:
                if not self.is_image_set:
                    raise ValueError("image must be set when image not in inputs")
                original_size = self.original_size
                model_inputs["image"] = None
                model_inputs["features"] = self.features
                model_inputs["original_size"] = self.original_size
                model_inputs["input_size"] = self.input_size

            point_coords = inputs.get("points", None)
            point_labels = inputs.get("labels", None)
            boxes = inputs.get("boxes", None)
            masks = inputs.get("masks", None)

            point_coords, point_labels, boxes, masks = self.preprocess_prompts(
                original_size=original_size,
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=boxes,
                masks=masks
            )

            model_inputs["point_coords"] = point_coords
            model_inputs["point_labels"] = point_labels
            model_inputs["boxes"] = boxes
            model_inputs["mask_inputs"] = masks
        else:
            image = inputs.get("image", None)
            if image is None:
                raise ValueError("An image must be supported, when 'seg_image' is True.")

            if isinstance(image, str):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_size = image.shape[:2]

            reset_config = preprocess_params.get("reset_config", False)
            if reset_config:
                self.reset_config(**preprocess_params)

            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, self.crop_n_layers, self.crop_overlap_ratio
            )
            model_inputs["original_size"] = orig_size

            model_inputs["image_list"] = []
            model_inputs["features_list"] = []
            model_inputs["original_size_list"] = []
            model_inputs["input_size_list"] = []
            model_inputs["crop_boxes_list"] = []
            model_inputs["points_for_image_list"] = []
            model_inputs["point_coords_list"] = []
            model_inputs["point_labels_list"] = []
            for crop_box, layer_id in zip(crop_boxes, layer_idxs):
                x0, y0, x1, y1 = crop_box
                crop_image = image[y0:y1, x0:x1, :]
                crop_image_size = crop_image.shape[:2]
                self.set_image(crop_image)

                points_scale = np.array(crop_image_size)[None, ::-1]
                points_for_image = self.point_grids[layer_id] * points_scale
                point_coords, point_labels, _, _ = self.preprocess_prompts(
                    original_size=crop_image_size,
                    point_coords=points_for_image,
                    multi_seg=True
                )

                model_inputs["features_list"].append(self.features)
                model_inputs["original_size_list"].append(self.original_size)
                model_inputs["input_size_list"].append(self.input_size)
                model_inputs["crop_boxes_list"].append(crop_box)
                model_inputs["points_for_image_list"].append(points_for_image)
                model_inputs["point_coords_list"].append(point_coords)
                model_inputs["point_labels_list"].append(point_labels)

        return model_inputs

    def _forward(self, model_inputs, **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (dict):
                The output of preprocess.
            forward_params (dict):
                The parameter dict for model forward.
        """
        multimask_output = forward_params.get("multimask_output", True)
        return_logits = forward_params.get("return_logits", False)

        seg_image = forward_params["seg_image"]
        if not seg_image:
            masks, low_res_masks, iou_predictions = self.model(
                image=model_inputs["image"],
                features=model_inputs["features"],
                input_size=model_inputs["input_size"],
                original_size=model_inputs["original_size"],
                point_coords=model_inputs["point_coords"],
                point_labels=model_inputs["point_labels"],
                boxes=model_inputs["boxes"],
                mask_inputs=model_inputs["mask_inputs"],
                multimask_output=multimask_output,
                return_logits=return_logits
            )
            return {"masks": masks, "low_res_masks": low_res_masks, "iou_predictions": iou_predictions}

        orig_size = model_inputs["original_size"]
        orig_h, orig_w = orig_size

        points_for_image_list = model_inputs["points_for_image_list"]
        point_coords_list = model_inputs["point_coords_list"]
        point_labels_list = model_inputs["point_labels_list"]
        crop_boxes_list = model_inputs["crop_boxes_list"]

        data = MaskData()
        for layer_id, features in enumerate(model_inputs["features_list"]):
            point_coords = point_coords_list[layer_id]
            point_labels = point_labels_list[layer_id]
            points_for_image = points_for_image_list[layer_id]
            crop_box = crop_boxes_list[layer_id]

            input_size = model_inputs["input_size_list"][layer_id]
            original_size = model_inputs["original_size_list"][layer_id]
            # Generate masks for this crop in batches
            data_crop = MaskData()
            for (points_for_image_batch, point_coords_batch, point_labels_batch) in \
                batch_iterator(self.points_per_batch, points_for_image, point_coords, point_labels):
                masks, low_res_masks, iou_predictions = self.model(
                    features=features,
                    input_size=input_size,
                    original_size=original_size,
                    point_coords=point_coords_batch,
                    point_labels=point_labels_batch,
                    multimask_output=multimask_output,
                    return_logits=True
                )

                masks = masks.asnumpy()
                low_res_masks = low_res_masks.asnumpy()
                iou_predictions = iou_predictions.asnumpy()

                data_batch = MaskData(
                    masks=masks.reshape((-1, masks.shape[-2], masks.shape[-1])),
                    iou_preds=iou_predictions.reshape((-1,)),
                    points=points_for_image_batch.repeat(masks.shape[1], axis=0)
                )
                del masks

                # Filter by predicted IoU
                if self.pred_iou_thresh > 0.0:
                    keep_mask = data_batch["iou_preds"] > self.pred_iou_thresh
                    data_batch.filter(keep_mask)

                # Calculate stability score
                data_batch["stability_score"] = calculate_stability_score(
                    data_batch["masks"], self.model.mask_threshold, self.stability_score_offset
                )
                if self.stability_score_thresh > 0.0:
                    keep_mask = data_batch["stability_score"] >= self.stability_score_thresh
                    data_batch.filter(keep_mask)

                # Threshold masks and calculate boxes
                data_batch["masks"] = data_batch["masks"] > self.model.mask_threshold
                data_batch["boxes"] = batched_mask_to_box(data_batch["masks"])

                # Filter boxes that touch crop boundaries
                keep_mask = ~is_box_near_crop_edge(data_batch["boxes"], crop_box, [0, 0, orig_w, orig_h])
                if not keep_mask.all():
                    data_batch.filter(keep_mask)

                # Compress to RLE
                data_batch["masks"] = uncrop_masks(data_batch["masks"], crop_box, orig_h, orig_w)
                data_batch["rles"] = mask_to_rle(data_batch["masks"])
                del data_batch["masks"]

                data_crop.cat(data_batch)

            # Remove duplicates within this crop.
            input_boxes = data_crop["boxes"]
            input_scores = data_crop["iou_preds"]
            keep_by_nms = nms(input_boxes, input_scores, iou_threshold=self.crop_nms_thresh)
            data_crop.filter(keep_by_nms)

            # Return to the original image frame
            data_crop["boxes"] = uncrop_boxes_xyxy(data_crop["boxes"], crop_box)
            data_crop["points"] = uncrop_points(data_crop["points"], crop_box)
            data_crop["crop_boxes"] = np.array([crop_box for _ in range(len(data_crop["rles"]))])

            data.cat(data_crop)

        # Remove duplicate masks between crops
        if len(crop_boxes_list) > 1:
            # Prefer masks from smaller crops
            input_boxes = data["boxes"]
            input_scores = 1.0 / box_area(data["crop_boxes"])
            keep_by_nms = nms(input_boxes, input_scores, iou_threshold=self.crop_nms_thresh)
            data.filter(keep_by_nms)

        return data

    def postprocess(self, model_outputs, **postprocess_params):
        r"""Postprocess

        Args:
            model_outputs (dict):
                Outputs of forward process.
            postprocess_params (dict):
                The parameter dict for postprocess.

        Return:
            classification results.
        """
        seg_image = postprocess_params["seg_image"]

        if not seg_image:
            bs = model_outputs["masks"].shape[0]
            if bs == 1:
                for k, v in model_outputs.items():
                    model_outputs[k] = v[0].asnumpy()
            else:
                for k, v in model_outputs.items():
                    model_outputs[k] = v.asnumpy()

            return model_outputs

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            model_outputs = self.postprocess_small_regions(
                model_outputs,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            model_outputs["segmentations"] = [coco_encode_rle(rle) for rle in model_outputs["rles"]]
        elif self.output_mode == "binary_mask":
            model_outputs["segmentations"] = [rle_to_mask(rle) for rle in model_outputs["rles"]]
        else:
            model_outputs["segmentations"] = model_outputs["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(model_outputs["segmentations"])):
            ann = {
                "segmentation": model_outputs["segmentations"][idx],
                "area": area_from_rle(model_outputs["rles"][idx]),
                "bbox": box_xyxy_to_xywh(model_outputs["boxes"][idx]).tolist(),
                "predicted_iou": model_outputs["iou_preds"][idx].item(),
                "point_coords": [model_outputs["points"][idx].tolist()],
                "stability_score": model_outputs["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(model_outputs["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def preprocess_image(self,
                         image,
                         image_format: str = "RGB") -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Args:
            image (np.ndarray):
                The image for calculating masks. Expects an image in HWC uint8 format, with pixel values in [0, 255].
            image_format (str):
                The color format of the image, in ['RGB', 'BGR'].
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image_format not in ["RGB", "BGR"]:
            raise ValueError(f"image_format must be in ['RGB', 'BGR'], but got {image_format}.")
        if image_format != self.network.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        original_size = image.shape[:2] # h, w
        input_image, input_size = self.image_processor(image)

        return input_image, original_size, input_size

    def preprocess_prompts(self,
                           original_size,
                           point_coords=None,
                           point_labels=None,
                           boxes=None,
                           masks=None,
                           multi_seg=False):
        """preprocess prompts"""
        if point_coords is not None:
            ndim = point_coords.ndim
            if ndim == 1:
                point_coords = point_coords[None, :]
            if ndim == 2:
                if point_labels is None:
                    point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
                point_coords = self.image_processor.transform.apply_coords(point_coords, original_size)
                point_coords = np.expand_dims(point_coords, axis=0)
                point_labels = np.expand_dims(point_labels, axis=0)
                if multi_seg:
                    point_coords = point_coords.transpose(1, 0, 2)
                    point_labels = point_labels.transpose()
            elif ndim == 3:
                if point_labels is None:
                    point_labels = np.ones((point_coords.shape[0], point_coords.shape[1]), dtype=np.int32)
                point_coords = self.image_processor.transform.apply_coords_batch(point_coords, original_size)
            else:
                raise ValueError("points's ndim < 3.")
            point_coords = Tensor(point_coords, dtype=ms.float32)
            point_labels = Tensor(point_labels, dtype=ms.int32)

        if boxes is not None:
            ndim = boxes.ndim
            if ndim == 1:
                boxes = self.image_processor.transform.apply_boxes(boxes, original_size)
                boxes = np.expand_dims(boxes, axis=0)
            elif ndim == 2:
                boxes = self.image_processor.transform.apply_boxes_batch(boxes, original_size)
            else:
                raise ValueError("boxes's ndim must be 1 or 2.")
            boxes = Tensor(boxes, dtype=ms.float32)

        if masks is not None:
            masks = np.expand_dims(masks, axis=0)
            masks = Tensor(masks, dtype=ms.float32)

        return point_coords, point_labels, boxes, masks

    def set_image(self,
                  image,
                  image_format: str = "RGB"):
        self.reset_image()
        input_image, original_size, input_size = self.preprocess_image(image, image_format)
        self.features = self.network.image_encoder(input_image)
        self.original_size = original_size # h, w
        self.input_size = input_size # h, w
        self.is_image_set = True

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.input_size = None

    def get_image_embedding(self):
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) to generate an embedding.")
        if self.features is None:
            raise ValueError("self.features should not be None.")
        return self.features

    def reset_config(self,
                     points_per_side=32,
                     points_per_batch=64,
                     pred_iou_thresh=0.88,
                     stability_score_thresh=0.95,
                     stability_score_offset=1.0,
                     box_nms_thresh=0.7,
                     crop_n_layers=0,
                     crop_nms_thresh=0.7,
                     crop_overlap_ratio=0.3413,
                     crop_n_points_downscale_factor=1,
                     point_grids=None,
                     min_mask_region_area=0,
                     output_mode="binary_mask",
                     **kwargs) -> None:
        """reset config"""
        self.points_per_side = points_per_side or kwargs.get('points_per_side', 32)
        self.points_per_batch = points_per_batch or kwargs.get('points_per_batch', 64)
        self.pred_iou_thresh = pred_iou_thresh or kwargs.get('points_per_batch', 0.88)
        self.stability_score_thresh = stability_score_thresh or kwargs.get('stability_score_thresh', 0.95)
        self.stability_score_offset = stability_score_offset or kwargs.get('stability_score_offset', 1.0)
        self.box_nms_thresh = box_nms_thresh or kwargs.get('box_nms_thresh', 0.7)
        self.crop_n_layers = crop_n_layers or kwargs.get('crop_n_layers', 0)
        self.crop_nms_thresh = crop_nms_thresh or kwargs.get('crop_nms_thresh', 0.7)
        self.crop_overlap_ratio = crop_overlap_ratio or kwargs.get('crop_overlap_ratio', 0.3413)
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor or\
                                              kwargs.get('crop_n_points_downscale_factor', 1)
        self.point_grids = point_grids or kwargs.get('point_grids', None)
        self.min_mask_region_area = min_mask_region_area or kwargs.get('min_mask_region_area', 0)
        self.output_mode = output_mode or kwargs.get('output_mode', "binary_mask")

        if (self.points_per_side is None) == (self.point_grids is None):
            raise ValueError("Exactly one of points_per_side or point_grid must be provided.")
        if self.points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                self.points_per_side,
                self.crop_n_layers,
                self.crop_n_points_downscale_factor
            )

        if self.output_mode not in ["binary_mask", "uncompressed_rle", "coco_rle"]:
            raise ValueError(f"Unknown output_mode: {self.output_mode}.")

    def postprocess_small_regions(self,
                                  mask_data: MaskData,
                                  min_area: int,
                                  nms_thresh: float) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if not mask_data["rles"]:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(np.expand_dims(mask, axis=0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = np.concatenate(new_masks, axis=0)
        boxes = batched_mask_to_box(masks)
        input_boxes = boxes.astype(np.float32)
        input_scores = np.array(scores)
        keep_by_nms = nms(input_boxes, input_scores, iou_threshold=nms_thresh)

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_tensor = np.expand_dims(masks[i_mask], axis=0)
                mask_data["rles"][i_mask] = mask_to_rle(mask_tensor)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
