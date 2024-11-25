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
# ============================================================================

"""models init"""
from .auto import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskGeneration,
    AutoModelForMaskedImageModeling,
    AutoModelForMultipleChoice,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTextEncoding,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelWithLMHead,
    AutoProcessor,
    AutoTokenizer,
    CONFIG_MAPPING,
    CONFIG_MAPPING_NAMES,
    IMAGE_PROCESSOR_MAPPING,
    ImageProcessingMixin,
    TOKENIZER_MAPPING
)
from .bert import (
    BasicTokenizer,
    BertConfig,
    BertForMultipleChoice,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForTokenClassification,
    BertModel,
    BertProcessor,
    BertTokenizer,
    BertTokenizerFast
)
from .mae import (
    ViTMAEConfig,
    ViTMAEForPreTraining,
    ViTMAEImageProcessor,
    ViTMAEModel,
    ViTMAEProcessor
)
from .vit import (
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
    ViTModel,
    ViTProcessor
)
from .swin import (
    SwinConfig,
    SwinForImageClassification,
    SwinImageProcessor,
    SwinModel,
    SwinProcessor
)
from .clip import (
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTextConfig,
    CLIPTokenizer,
    CLIPVisionConfig
)
from .t5 import (
    MT5ForConditionalGeneration,
    T5Config,
    T5ForConditionalGeneration,
    T5PegasusTokenizer,
    T5Processor,
    T5Tokenizer,
    T5TokenizerFast
)
from .gpt2 import (
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2Processor,
    GPT2Tokenizer,
    GPT2TokenizerFast
)
from .glm import (
    ChatGLMTokenizer,
    GLMChatModel,
    GLMConfig,
    GLMForPreTraining,
    GLMProcessor
)
from .glm2 import (
    ChatGLM2Config,
    ChatGLM2ForConditionalGeneration,
    ChatGLM2Model,
    ChatGLM2Tokenizer,
    ChatGLM2WithPtuning2,
    ChatGLM3Tokenizer,
    ChatGLM4Tokenizer
)
from .llama import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaProcessor,
    LlamaTokenizer,
    LlamaTokenizerFast
)
from .pangualpha import (
    PanguAlphaConfig,
    PanguAlphaHeadModel,
    PanguAlphaModel,
    PanguAlphaProcessor,
    PanguAlphaPromptTextClassificationModel,
    PanguAlphaTokenizer
)
from .bloom import (
    BloomConfig,
    BloomLMHeadModel,
    BloomModel,
    BloomProcessor,
    BloomRewardModel,
    BloomTokenizer,
    BloomTokenizerFast,
    VHead
)
from .sam import (
    ImageEncoderConfig,
    MaskData,
    SamConfig,
    SamImageEncoder,
    SamImageProcessor,
    SamMaskDecoder,
    SamModel,
    SamProcessor,
    SamPromptEncoder,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_area,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle,
    nms,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points
)
from .cogvlm2 import (
    CogVLM2Config,
    CogVLM2ContentTransformTemplate,
    CogVLM2ForCausalLM,
    CogVLM2ImageContentTransformTemplate,
    CogVLM2ImageForCausalLM,
    CogVLM2Tokenizer,
    CogVLM2VideoLM,
    CogVLM2VideoLMModel,
    LlamaForCausalLMForCogVLM2Image
)
from .eva02 import (
    EVA02Config,
    EVAModel
)
from .whisper import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperTokenizer
)
from .tokenization_utils import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase
)
from .tokenization_utils_fast import PreTrainedTokenizerFast
from .modeling_utils import PreTrainedModel
from .multi_modal import (
    BaseImageToTextImageProcessor,
    BaseTextContentBuilder,
    BaseXModalToTextModel,
    BaseXModalToTextProcessor,
    BaseXModalToTextTransform,
    ModalContentTransformTemplate
)
from .configuration_utils import PretrainedConfig
from .base_config import BaseConfig
from .base_model import BaseModel
from .image_processing_utils import BaseImageProcessor
from .processing_utils import ProcessorMixin
from .base_processor import (
    BaseAudioProcessor,
    BaseProcessor
)
from .build_tokenizer import build_tokenizer
from .build_processor import build_processor
from .build_model import (
    build_encoder,
    build_head,
    build_model,
    build_model_config,
    build_network
)
from .utils import (
    CONFIG_NAME,
    FEATURE_EXTRACTOR_NAME,
    IMAGE_PROCESSOR_NAME
)

__all__ = ['PreTrainedTokenizer', 'PreTrainedTokenizerFast']
__all__.extend(auto.__all__)
__all__.extend(glm2.__all__)
__all__.extend(llama.__all__)
__all__.extend(multi_modal.__all__)
__all__.extend(configuration_utils.__all__)
__all__.extend(modeling_utils.__all__)
