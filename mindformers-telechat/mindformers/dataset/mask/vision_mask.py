# Copyright 2022 Huawei Technologies Co., Ltd
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
# This file was refer to project:
# https://github.com/adapter-hub/adapter-transformers/blob/master/examples/pytorch/image-pretraining/run_mim.py
# ============================================================================
"""Self-Define Vision Mask Policy."""
import numpy as np
from mindspore.dataset.transforms import py_transforms
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['SimMask', 'MaeMask']


@MindFormerRegister.register(MindFormerModuleType.MASK_POLICY)
class SimMask(py_transforms.PyTensorOperation):
    """SimMIM Mask Policy."""
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        super(SimMask, self).__init__()
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        if self.input_size % self.mask_patch_size != 0:
            raise ValueError(f"input_size % mask_patch_size must be 0, but get input_size {self.input_size} and "
                             f"mask_patch_size {self.mask_patch_size}")
        if self.mask_patch_size % self.model_patch_size != 0:
            raise ValueError(f"mask_patch_size % model_patch_size must be 0, but get mask_patch_size "
                             f"{self.mask_patch_size} and model_patch_size {self.model_patch_size}")

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=np.int32)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        return img, mask

    def __repr__(self):
        return "Mask generator for simmin arch."


@MindFormerRegister.register(MindFormerModuleType.MASK_POLICY)
class MaeMask(py_transforms.PyTensorOperation):
    """MAE Mask Policy."""
    def __init__(self, input_size=192, patch_size=4, mask_ratio=0.75):
        super(MaeMask, self).__init__()
        if not 0 < mask_ratio < 1:
            raise ValueError('masking ratio must be kept between 0 and 1, but get mask_ratio {mask_ratio}.')
        # seq_length
        self.num_patches = (input_size // patch_size) ** 2
        # seq masked number
        self.keep_num = int((1 - mask_ratio) * self.num_patches)

    def __call__(self, imgs):
        rand_indices = np.argsort(
            np.random.uniform(size=(self.num_patches,)), axis=0).astype(np.int32)
        ids_restore = np.argsort(rand_indices, axis=0).astype(np.int32)
        mask = np.ones((self.num_patches,)).astype(np.int32)
        mask[:self.keep_num] = 0
        unmask_index = rand_indices[:self.keep_num]
        out = (imgs, mask, ids_restore, unmask_index,)
        return out

    def __repr__(self):
        return "Mask generator for mae arch."
