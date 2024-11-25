# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
CogVLM2Processor
"""
from typing import Optional, Union, List, Dict, Any
import PIL
import PIL.Image
import cv2
import numpy as np
import mindspore as ms

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.dataset.transforms.vision_transforms import BatchToTensor, BatchCenterCrop
from mindformers.models.image_processing_utils import BaseImageProcessor

from ..multi_modal import ModalContentTransformTemplate, BaseTextContentBuilder
from ..multi_modal.base_multi_modal_processor import BatchResizeV2
from ..multi_modal.modal_content import ModalContentBuilder
from ..multi_modal.utils import DataRecord

video_format_list = ('.mp4', '.avi', '.flv', '.mpeg', '.f4v', '.mkv')


class VideoProcessor:
    """
    Process input video data with opencv-python.
    """

    def __init__(self,
                 num_frames: int = 24,
                 pad_frames: bool = False,
                 frames_interval: Union[int, str] = 1):
        self.num_frames = num_frames
        self.pad_frames = pad_frames
        self.frames_interval = frames_interval

    def __call__(self, video_file_path):
        if not isinstance(video_file_path, str):
            video_file_path = str(video_file_path)
        frames = self.cv_video_to_image(video_file_path)
        return frames

    # pylint: disable=W0640
    @staticmethod
    def get_frame_id(fps, frame_num, frame_interval, max_num):
        """Get frame id for different strategy."""
        seconds = [_ / fps for _ in range(frame_num)]
        max_second = round(max(seconds)) + 1
        sec_idx = []
        for sec in range(max_second):
            closest_num = min(seconds, key=lambda x: abs(x - sec))
            index = seconds.index(closest_num)
            sec_idx.append(index)
        frame_ids = []
        for idx in range(len(sec_idx) - 1):
            start_idx = sec_idx[idx]
            end_idx = sec_idx[idx + 1]
            for _ in range(frame_interval):
                fetch_idx = (_ / frame_interval) * (end_idx - start_idx) + start_idx
                frame_ids.append(fetch_idx)
                if len(frame_ids) >= max_num:
                    return frame_ids
        return frame_ids

    def cv_video_to_image(self, video_path):
        """Extract input video frames with cv2."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if isinstance(self.frames_interval, int) and self.frames_interval > 0:
            frame_ids = self.get_frame_id(fps, frame_num, self.frames_interval, self.num_frames)
        elif self.frames_interval == 'average':
            frame_ids = np.linspace(0, frame_num - 1, self.num_frames, dtype=np.int32)
        else:
            raise ValueError(f"unsupported video frames_interval strategy: {self.frames_interval}.")

        frames = []
        for idx in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError(f"frames in video may corrupt, please check video file {video_path}.")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < self.num_frames and self.pad_frames:
            last_frame = frames[-1]
            frames += [last_frame for _ in range(self.num_frames - len(frames))]

        return np.array(frames)


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class CogVLM2ImageProcessor(BaseImageProcessor):
    """
    CogVLM2 ImageProcessor.

    Args:
        image_size (int): The target size.
    """

    def __init__(self,
                 image_size: Optional[int] = 224,
                 interpolation: Optional[str] = 'linear',
                 mean: Optional[list] = None,
                 std: Optional[list] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.resize = BatchResizeV2(image_size, interpolation=interpolation)
        self.to_tensor = BatchToTensor()
        if mean is None:
            mean = [0.48145466, 0.4578275, 0.40821073]
        self.mean = np.array(mean)
        if std is None:
            std = [0.26862954, 0.26130258, 0.27577711]
        self.std = np.array(std)
        self.center_crop = BatchCenterCrop(image_size)

    def preprocess(self, images: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]], **kwargs):
        r"""
        Preprocess Required By Base Processor.

        Args:
            images (numpy.ndarray): A batch of images.

        Return:
            A 4-rank tensor for a batch of images.
        """
        images = images / 255.0
        images = (images - self.mean) / self.std
        images = self.resize(images)
        images = self.center_crop(images)
        images = np.transpose(images.astype(np.float32), (0, 3, 1, 2))
        return ms.Tensor(images)

    @staticmethod
    def _bhwc_check(image_batch: Union[ms.Tensor, PIL.Image.Image, np.ndarray, List[PIL.Image.Image]]):
        r"""Bhwc_check"""
        if isinstance(image_batch, np.ndarray):
            if image_batch.shape[-1] == 3:
                return True
        if isinstance(image_batch, ms.Tensor):
            if image_batch.asnumpy().shape[-1] == 3:
                return True
        if isinstance(image_batch, (list, PIL.Image.Image)):
            return True
        return False


class CogVLM2VideoContentBuilder(ModalContentBuilder):
    """
    CogVLM2 Video Content Builder.
    """

    def __init__(
            self,
            context_pad_token,
            context_length,
            use_custom_token: bool = True,
            start_token: str = None,
            end_token: str = None,
            tokenizer: Any = None,
            num_frames: int = 24,
            pad_frames: bool = False,
            image_size: int = 224,
            frames_interval: Union[int, str] = 1,
            is_train: bool = False
    ):
        super(CogVLM2VideoContentBuilder, self).__init__(
            type_="video",
            context_pad_token=context_pad_token,
            context_length=context_length,
            use_custom_token=use_custom_token,
            start_token=start_token,
            end_token=end_token,
            tokenizer=tokenizer
        )
        self.vision_token_type = 1
        self.language_token_type = 0
        self.is_train = is_train
        self.video_reader = VideoProcessor(num_frames=num_frames,
                                           pad_frames=pad_frames,
                                           frames_interval=frames_interval)
        self.image_processor = CogVLM2ImageProcessor(image_size)

    # pylint: disable=W0613
    def regular_input_for_predict(self, inputs: Dict, result_recorder: DataRecord = None, **kwargs):
        """regular input for predict."""
        video_path = inputs["video"]
        images = self.video_reader(video_path)
        images = self.image_processor(images)
        text = self.start_token + self.end_token
        result_recorder.put("images", images)
        result_recorder.put('frame_num', images.shape[0])
        return text

    def regular_input_for_train(self, inputs, result_recorder: DataRecord = None, **kwargs):
        """regular input for train."""
        text, video_path = self.find_tag_and_extract(inputs, self.start_token, self.end_token)
        if len(video_path) != 1:
            raise ValueError(f"number of video path in text should be 1, but got {len(video_path)}")
        images = self.video_reader(video_path[0])
        images = self.image_processor(images)
        result_recorder.put("images", images)
        result_recorder.put('frame_num', images.shape[0])
        return text

    # pylint: disable=W0613
    def padding_context(self, input_ids, result_recorder: DataRecord = None, **kwargs):
        """pad text input with pad_token_id."""
        start_position = np.where(input_ids == self.start_token_id)[0]

        if start_position.size == 0:
            return input_ids

        context_input_ids = [self.tokenizer.bos_token_id]
        token_type_ids = [self.language_token_type]
        position_ids = [0]

        video_token_ids = []
        types = []
        cur_position = 1
        for time_index in range(result_recorder.get('frame_num')):
            video_token_ids += [self.context_pad_token_id] * self.context_length
            types += [self.vision_token_type] * self.context_length
            if not self.is_train:
                time_indices = self.tokenizer(str(time_index), add_special_tokens=False)["input_ids"]
                len_time_indices = len(time_indices)

                video_token_ids += time_indices
                types += [self.language_token_type] * len_time_indices

                position_ids += [cur_position] + [cur_position + 1] * (self.context_length - 2) + \
                                [cur_position + 2] + [cur_position + 3 + i for i in range(len_time_indices)]
                cur_position += 3 + len_time_indices
            else:
                # do not insert frame time index in position_ids
                if position_ids[-1] == self.language_token_type:
                    position_ids += [cur_position]
                    cur_position += 1
                    position_ids += [cur_position] * (self.context_length - 1)
                else:
                    position_ids += [cur_position] * self.context_length
        if self.is_train:
            position_ids[-1] = cur_position + 1
            cur_position += 2

        context_input_ids += video_token_ids
        token_type_ids += types

        input_ids = np.insert(input_ids, [start_position[0]], context_input_ids)

        input_ids = np.delete(input_ids, np.where(input_ids == self.start_token_id))
        input_ids = np.delete(input_ids, np.where(input_ids == self.end_token_id))

        position_ids = np.array(position_ids)
        padding_length = len(input_ids) - len(position_ids)
        position_ids = np.pad(position_ids, (0, padding_length), 'linear_ramp',
                              end_values=(0, cur_position + padding_length - 1))

        result_recorder.put("position_ids", position_ids)
        result_recorder.put("valid_position", np.array([position_ids.size]))
        return input_ids


@MindFormerRegister.register(MindFormerModuleType.TRANSFORMS)
class CogVLM2ContentTransformTemplate(ModalContentTransformTemplate):
    """
    CogVLM2 Modal Content Transform Template.
    """

    # pylint: disable=W1113
    def __init__(self, output_columns, tokenizer, mode="predict", signal_type="base", *args, **kwargs):
        super().__init__(output_columns=output_columns, tokenizer=tokenizer, mode=mode, *args, **kwargs)
        self.signal_type = signal_type
        self.video_pad_token = "<|reserved_special_token_2|>"
        self.video_start_token = "<|reserved_special_token_3|>"
        self.video_end_token = "<|reserved_special_token_4|>"
        if mode == 'predict':
            frames_interval = 1
            is_train = False
        else:
            frames_interval = 'average'
            is_train = True

        self.modal_builders = {
            "video": CogVLM2VideoContentBuilder(
                self.video_pad_token, 66, use_custom_token=False,
                start_token=self.video_start_token, end_token=self.video_end_token,
                frames_interval=frames_interval, is_train=is_train),
            "text": BaseTextContentBuilder(),
        }

    # pylint: disable=W0613
    def build_conversation_input_text(self, raw_inputs, result_recorder):
        """Build conversion input text."""
        if self.signal_type == "base":
            return "".join(raw_inputs)
        if self.signal_type == "vqa":
            answer_format = "Short answer:"
        elif self.signal_type == "chat":
            answer_format = "Answer:"
        else:
            raise ValueError(f"Unknown signal type {self.signal_type}")

        if self.mode == 'train':
            return self.build_train_conversation(raw_inputs, answer_format)
        return self.build_predict_conversation(raw_inputs, answer_format)

    def build_train_conversation(self, raw_inputs, answer_format):
        """Build conversion for training."""
        self.history = raw_inputs[:-1]
        prompt = f"{self.video_start_token}{self.video_end_token}"
        for _, (key_from, value) in enumerate(self.history):
            if key_from == 'user':
                cur_value = value.replace(f"{self.video_start_token}{self.video_end_token}", '')
                prompt += f"Question: {cur_value} "
            else:
                prompt += f"{answer_format} {value}\n"
        text_list = [f"{prompt}{answer_format}", raw_inputs[-1][1], "<|end_of_text|>"]
        return text_list

    def build_predict_conversation(self, raw_inputs, answer_format):
        """Build conversion for prediction."""
        prompt = ""
        for _, (old_query, response) in enumerate(self.history):
            prompt += f"Question: {old_query} {answer_format} {response}\n"
        video_context_placeholder = (f"{self.modal_builders['video'].start_token}"
                                     f"{self.modal_builders['video'].end_token}")
        if video_context_placeholder in raw_inputs:
            video_query_index = raw_inputs.index(video_context_placeholder)
            video_query = raw_inputs.pop(video_query_index)
            prompt = f"{video_query}{prompt}"
        query = "".join(raw_inputs)
        prompt += f"Question: {query} {answer_format}"
        return prompt

    def get_need_update_output_items(self, result: DataRecord) -> Dict[str, Any]:
        """Update result before output."""
        update_items = {}
        if not result.has_key("position_ids"):
            position_ids = np.arange(len(result.get("input_ids")), dtype=np.int32)
            update_items["position_ids"] = ms.Tensor(position_ids)
        if self.mode == 'train':
            position_ids = result.get("position_ids")
            padding_start = position_ids[-1]
            padding_length = self.max_length - len(position_ids) + 1
            position_ids = np.pad(position_ids, (0, padding_length), 'linear_ramp',
                                  end_values=(0, padding_start + padding_length))
            update_items["position_ids"] = position_ids
            images = result.get("images")
            if isinstance(images, ms.Tensor):
                images = images.numpy()
            update_items["images"] = images
        return update_items

    def batch(self, data_list, token_padding_length, **kwargs):
        """Get batched input data."""
        batched_data = super().batch(data_list, token_padding_length, **kwargs)

        position_ids_list = batched_data.get("position_ids")
        padded_position_ids_list = []
        for position_ids in position_ids_list:
            padding_length = token_padding_length - len(position_ids)
            if padding_length > 0:
                padding_start = position_ids[-1]
                position_ids = np.pad(position_ids, (0, padding_length), 'linear_ramp',
                                      end_values=(0, padding_start + padding_length))
            else:
                position_ids = position_ids[:token_padding_length]

            padded_position_ids_list.append(position_ids)
        batched_data["position_ids"] = ms.Tensor(np.stack(padded_position_ids_list, axis=0), ms.int32)
        return batched_data

    # pylint: disable=W0613
    def post_process(self, output_ids, **kwargs):
        """Post process model output."""
        output = []
        for output_ids_item in output_ids:
            decoded = self.tokenizer.decode(output_ids_item, skip_special_tokens=True)
            if self.signal_type == "vqa":
                processed = decoded.split("Short answer:")[-1]
            elif self.signal_type == "chat":
                processed = decoded.split("Answer:")[-1]
            else:
                processed = decoded
            output.append(processed)
        return output

    def build_labels(self, text_id_list, result_recorder: DataRecord, **kwargs):
        """Build labels for cogvlm2 video."""
        text_id, label_id, eos_token = text_id_list
        labels = np.array([-100] * len(text_id))
        labels = np.concatenate((labels, label_id, eos_token))
        return labels
