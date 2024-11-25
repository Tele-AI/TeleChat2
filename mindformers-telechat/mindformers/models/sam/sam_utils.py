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
"""SAM Utils"""
import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple
import cv2
import numpy as np

__all__ = [
    "MaskData",
    "area_from_rle",
    "batch_iterator",
    "batched_mask_to_box",
    "box_xyxy_to_xywh",
    "build_all_layer_point_grids",
    "calculate_stability_score",
    "coco_encode_rle",
    "generate_crop_boxes",
    "is_box_near_crop_edge",
    "mask_to_rle",
    "remove_small_regions",
    "rle_to_mask",
    "uncrop_boxes_xyxy",
    "uncrop_masks",
    "uncrop_points",
    "box_area",
    "nms"
]


class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the MaskData.

        Args:
            **kwargs: Keyword arguments representing different mask-related data.
        """
        for v in kwargs.values():
            if not isinstance(v, (list, np.ndarray)):
                raise ValueError("MaskData only supports list, numpy arrays, and ms tensors.")
        self._stats = dict(**kwargs)

    def __setitem__(self, key: str, item: Any) -> None:
        """
        Set an item in the MaskData.

        Args:
            key (str): The key of the item.
            item (Any): The item to be set.
        """
        if not isinstance(item, (list, np.ndarray)):
            raise ValueError("MaskData only supports list, numpy arrays.")
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        """
        Delete an item from the MaskData.

        Args:
            key (str): The key of the item to be deleted.
        """
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        """
        Get an item from the MaskData.

        Args:
            key (str): The key of the item to be retrieved.

        Returns:
            Any: The retrieved item.
        """
        return self._stats[key]

    def items(self) -> ItemsView[str, Any]:
        """
        Get the items stored in the MaskData.

        Returns:
            ItemsView[str, Any]: A view of the items in the MaskData.
        """
        return self._stats.items()

    def filter(self, keep: np.ndarray) -> None:
        """
        Filter the MaskData based on a boolean mask.

        Args:
            keep (np.ndarray): Boolean mask indicating which items to keep.
        """
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep]
            elif isinstance(v, list) and keep.dtype == np.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        """
        Concatenate new data from another MaskData instance.

        Args:
            new_stats (MaskData): The MaskData instance to concatenate.
        """
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")


def is_box_near_crop_edge(boxes: np.ndarray,
                          crop_box: List[int],
                          orig_box: List[int],
                          atol: float = 20.0) -> np.ndarray:
    """
    Filter boxes at the edge of a crop, but not at the edge of the original image.

    Args:
        boxes (np.ndarray): Bounding boxes ndarray of shape (N, 4) in format (x1, y1, x2, y2).
        crop_box (List[int]): Crop box coordinates in format (x1, y1, x2, y2).
        orig_box (List[int]): Original image box coordinates in format (x1, y1, x2, y2).
        atol (float): Absolute tolerance for box comparison.

    Returns:
        np.ndarray: Boolean ndarray indicating whether boxes are near the crop edge.
    """
    crop_box = np.array(crop_box, dtype=np.float32)
    orig_box = np.array(orig_box, dtype=np.float32)
    boxes = uncrop_boxes_xyxy(boxes, crop_box).astype(np.float32)

    near_crop_edge = np.isclose(boxes,
                                np.tile(crop_box[None, :], (boxes.shape[0], 1)),
                                atol=atol,
                                rtol=0.,
                                equal_nan=True)
    near_image_edge = np.isclose(boxes,
                                 np.tile(orig_box[None, :], (boxes.shape[0], 1)),
                                 atol=atol,
                                 rtol=0.,
                                 equal_nan=True)
    near_crop_edge = np.logical_and(near_crop_edge, ~near_image_edge)

    return np.any(near_crop_edge, axis=1)


def box_xyxy_to_xywh(box_xyxy: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, w, h) format.

    Args:
        box_xyxy (np.ndarray): Bounding boxes tensor of shape (N, 4) in (x1, y1, x2, y2) format.

    Returns:
        box_xywh (np.ndarray): Bounding boxes tensor of shape (N, 4) in (x, y, w, h) format.
    """
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """
    Generate batches of data from input arrays.

    Args:
        batch_size (int): Size of each batch.
        *args: Variable number of input arrays to iterate over.

    Yields:
        List[Any]: A batch of data containing corresponding elements from each input array.
    """
    if not args or not all(len(a) == len(args[0]) for a in args):
        raise ValueError("Batched iteration must have inputs of all the same size.")
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def mask_to_rle(tensor: np.ndarray) -> List[Dict[str, Any]]:
    """
    Encode masks to an uncompressed RLE format, in the format expected by
    pycoco tools.

    Args:
        tensor (np.ndarray): Binary mask tensor of shape (B, H, W) where B is the batch size.

    Returns:
        List[Dict[str, Any]]: List of dictionaries representing RLE encoded masks.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.transpose(0, 2, 1).reshape(tensor.shape[0], -1)

    # Compute change indices
    diff = np.bitwise_xor(tensor[:, 1:].astype(np.int32), tensor[:, :-1].astype(np.int32))
    change_indices = np.stack(np.nonzero(diff)).transpose()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = np.concatenate(
            [
                np.array([0], dtype=cur_idxs.dtype),
                cur_idxs + 1,
                np.array([h * w], dtype=cur_idxs.dtype)
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    Compute a binary mask from an uncompressed RLE.

    Args:
        rle (Dict[str, Any]): Dictionary containing RLE encoded mask information.

    Returns:
        np.ndarray: Binary mask as a NumPy array.
    """
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def area_from_rle(rle: Dict[str, Any]) -> int:
    """
    Calculate the area of a binary mask from an uncompressed RLE.

    Args:
        rle (Dict[str, Any]): Dictionary containing RLE encoded mask information.

    Returns:
        int: Calculated area of the mask.
    """
    return sum(rle["counts"][1::2])


def calculate_stability_score(masks: np.ndarray,
                              mask_threshold: float,
                              threshold_offset: float) -> np.ndarray:
    """
    Calculate the stability score for a batch of masks. The stability
    score is the Intersection over Union (IoU) between the binary masks
    obtained by thresholding the predicted mask logits at high and low values.

    Args:
        masks (np.ndarray): Predicted mask logits tensor of shape (B, H, W).
        mask_threshold (float): Threshold value for binary mask creation.
        threshold_offset (float): Offset to adjust the threshold.

    Returns:
        np.ndarray: Stability scores for each mask in the batch.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=np.float32)
        .sum(-1, dtype=np.float32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=np.float32)
        .sum(-1, dtype=np.float32)
    )
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """
    Generate a 2D grid of points evenly spaced in the [0, 1] x [0, 1] range.

    Args:
        n_per_side (int): Number of points per side of the grid.

    Returns:
        np.ndarray: 2D grid of points as a NumPy array.
    """
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def build_all_layer_point_grids(n_per_side: int,
                                n_layers: int,
                                scale_per_layer: int) -> List[np.ndarray]:
    """
    Generate point grids for all crop layers.

    Args:
        n_per_side (int): Number of points per side of the original grid.
        n_layers (int): Number of layers in the crop hierarchy.
        scale_per_layer (int): Scaling factor applied per layer.

    Returns:
        List[np.ndarray]: List of point grids for each crop layer as NumPy arrays.
    """
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def generate_crop_boxes(im_size: Tuple[int, ...],
                        n_layers: int,
                        overlap_ratio: float) -> Tuple[List[List[int]], List[int]]:
    """
    Generate a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.

    Args:
        im_size (Tuple[int, ...]): Size of the original image in (height, width).
        n_layers (int): Number of layers in the crop hierarchy.
        overlap_ratio (float): Overlap ratio for calculating crop box size.

    Returns:
        Tuple[List[List[int]], List[int]]: Tuple containing a list of crop boxes and
        a list of layer indices for each crop box.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def uncrop_boxes_xyxy(boxes: np.ndarray, crop_box: List[int]) -> np.ndarray:
    """
    Uncrop the coordinates of bounding boxes from a cropped region to the original image.

    Args:
        boxes (np.ndarray): Bounding box coordinates in XYXY format (x0, y0, x1, y1).
        crop_box (List[int]): Cropped region's bounding box in XYXY format (x0, y0, x1, y1).

    Returns:
        np.ndarray: Uncropped bounding box coordinates in XYXY format.
    """
    x0, y0, _, _ = crop_box
    offset = np.array([[x0, y0, x0, y0]], dtype=np.int32)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = np.expand_dims(offset, 1)
    return boxes + offset


def uncrop_points(points: np.ndarray, crop_box: List[int]) -> np.ndarray:
    """
    Uncrop the coordinates of points from a cropped region to the original image.

    Args:
        points (np.ndarray): Point coordinates in XY format (x, y).
        crop_box (List[int]): Cropped region's bounding box in XYXY format (x0, y0, x1, y1).

    Returns:
        np.ndarray: Uncropped point coordinates in XY format.
    """
    x0, y0, _, _ = crop_box
    offset = np.array([[x0, y0]], dtype=np.int32)
    # Check if points has a channel dimension
    if len(points.shape) == 3:
        offset = np.expand_dims(offset, 1)
    return points + offset


def uncrop_masks(masks: np.ndarray,
                 crop_box: List[int],
                 orig_h: int,
                 orig_w: int) -> np.ndarray:
    """
    Uncrop binary masks from a cropped region to the original image size.

    Args:
        masks (np.ndarray): Binary masks ndarray.
        crop_box (List[int]): Cropped region's bounding box in XYXY format (x0, y0, x1, y1).
        orig_h (int): Original height of the image.
        orig_w (int): Original width of the image.

    Returns:
        np.ndarray: Uncropped binary masks ndarray.
    """
    x0, y0, x1, y1 = crop_box
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    # Calculate pad widths
    pad_width = [(0, 0), (y0, pad_y - y0), (x0, pad_x - x0)]
    # Pad the masks
    return np.pad(masks, pad_width, mode='constant', constant_values=0)


def remove_small_regions(mask: np.ndarray,
                         area_thresh: float,
                         mode: str) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions or holes in a binary mask.

    Args:
        mask (np.ndarray): Binary mask.
        area_thresh (float): Threshold area for removing small regions.
        mode (str): Either "holes" or "islands" indicating the type of regions to remove.

    Returns:
        Tuple[np.ndarray, bool]: A tuple containing the modified mask and a boolean indicating if the mask was modified.
    """
    if mode not in ["holes", "islands"]:
        raise ValueError(f"mode should be one of holes or islands, but got {mode}.")
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if not small_regions:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if not fill_labels:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encode uncompressed RLE (Run-Length Encoding) to COCO format RLE.

    Args:
        uncompressed_rle (Dict[str, Any]): Uncompressed RLE dictionary.

    Returns:
        Dict[str, Any]: Encoded RLE in COCO format.
    """
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle


def batched_mask_to_box(masks):
    """
    Convert masks to bounding boxes in XYXY format. Return [0,0,0,0] for an empty mask.

    Args:
        masks (np.ndarray): Binary masks of shape C1xC2x...xHxW.

    Returns:
        np.ndarray: Bounding boxes in XYXY format of shape C1xC2x...x4.
    """
    # NumPy doesn't raise an error on empty inputs, so check for empty masks here
    if np.prod(masks.shape) == 0:
        return np.zeros((*masks.shape[:-2], 4), dtype=np.int32)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    else:
        masks = np.expand_dims(masks, 0)

    # Get top and bottom edges
    in_height = np.max(masks.astype(np.float32), axis=-1)
    in_height_coords = in_height * np.arange(h)[None, :]
    bottom_edges = np.max(in_height_coords, axis=-1)
    in_height_coords = in_height_coords + h * (~in_height.astype(np.bool))
    top_edges = np.min(in_height_coords, axis=-1)

    # Get left and right edges
    in_width = np.max(masks.astype(np.float32), axis=-2)
    in_width_coords = in_width * np.arange(w)[None, :]
    right_edges = np.max(in_width_coords, axis=-1)
    in_width_coords = in_width_coords + w * (~in_width.astype(np.bool))
    left_edges = np.min(in_width_coords, axis=-1)

    # If the mask is empty, the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = ((right_edges < left_edges).astype(np.int32) |\
                    (bottom_edges < top_edges).astype(np.int32)).astype(np.bool)
    out = np.stack([left_edges, top_edges, right_edges, bottom_edges], axis=-1)
    out = out * np.expand_dims(~empty_filter, axis=-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape((*shape[:-2], 4))
    else:
        out = out[0]

    return out.astype(np.int32)


def box_area(boxes):
    """
    Compute the area of a set of bounding boxes.

    Args:
        boxes (np.ndarray): Bounding boxes for which the area will be computed.
                            Expected shape is [N, 4], where N is the number of boxes.
                            Each box is specified by its coordinates (x1, y1, x2, y2)
                            in the format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        np.ndarray: The area for each box in an ndarray of shape [N].
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1: np.ndarray, box2: np.ndarray):
    """
    Calculate the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        box1 (np.ndarray): NumPy array representing the first set of bounding boxes
                           in format [x_min, y_min, x_max, y_max].
        box2 (np.ndarray): NumPy array representing the second set of bounding boxes
                           in the same format.

    Returns:
        iou (np.ndarray): NumPy array containing the IoU values for each pair of boxes.

    Examples:
        >>> box1 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        >>> box2 = np.array([[5, 5, 15, 15], [8, 8, 18, 18]])
        >>> iou = box_iou(box1, box2)
        >>> print(iou)
        [[0.11111111 0.        ]
        [1.         0.23076923]]
    """
    area1 = box_area(box1)
    area2 = box_area(box2)

    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """
    Apply Non-Maximum Suppression (NMS) algorithm to filter a set of bounding boxes,
    eliminating overlapping boxes based on the Intersection over Union (IoU) threshold.

    Args:
        boxes (np.ndarray): NumPy array containing bounding box coordinates,
                            with each row representing a box [x_min, y_min, x_max, y_max].
        scores (np.ndarray): NumPy array containing scores for each corresponding bounding box.
        iou_threshold (float): IoU (Intersection over Union) threshold to determine if boxes overlap.

    Returns:
        keep (np.ndarray): NumPy array containing indices of the retained bounding boxes.

    Examples:
        >>> boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [8, 8, 18, 18]])
        >>> scores = np.array([0.9, 0.75, 0.85])
        >>> iou_threshold = 0.5
        >>> keep = nms(boxes, scores, iou_threshold)
        >>> print(keep)
        [0 2]
    """
    idxs = scores.argsort()
    keep = []
    while idxs.size > 0:
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)
    return keep
