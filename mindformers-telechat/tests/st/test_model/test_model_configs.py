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
""" test model configs """
import unittest
from mindformers.models.auto.configuration_auto import AutoConfig
from mindformers.models.bert import BertForPreTraining, BertForQuestionAnswering, \
    BertForTokenClassification, BertForMultipleChoice
from mindformers.models.blip2 import Blip2Qformer, Blip2ItmEvaluator, \
    Blip2Classifier, Blip2ImageToTextGeneration, Blip2Llm
from mindformers.models.bloom import BloomLMHeadModel
from mindformers.models.clip import CLIPModel
from mindformers.models.glm import GLMForPreTraining, GLMChatModel
from mindformers.models.glm2 import ChatGLM2ForConditionalGeneration
from mindformers.models.gpt2 import GPT2LMHeadModel, GPT2ForSequenceClassification
from mindformers.models.llama import LlamaForCausalLM
from mindformers.models.mae import ViTMAEForPreTraining
from mindformers.models.pangualpha import PanguAlphaPromptTextClassificationModel, \
    PanguAlphaHeadModel
from mindformers.models.sam import SamModel
from mindformers.models.swin import SwinForImageClassification
from mindformers.models.t5 import T5ForConditionalGeneration
from mindformers.models.vit import ViTForImageClassification


class TestBertConfig(unittest.TestCase):
    """test bert config"""
    def test_init_model_for_pretraining_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/bert/run_bert_tiny_uncased.yaml")
        config.checkpoint_name_or_path = ''
        model = BertForPreTraining(config)

    def test_init_model_for_qa_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/qa/run_qa_bert_base_uncased.yaml")
        config.checkpoint_name_or_path = ''
        model = BertForQuestionAnswering(config)

    def test_init_model_for_token_classification_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/tokcls/run_tokcls_bert_base_chinese.yaml")
        config.checkpoint_name_or_path = ''
        model = BertForTokenClassification(config)

    def test_init_model_for_multiple_choice_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/txtcls/run_txtcls_bert_base_uncased.yaml")
        config.checkpoint_name_or_path = ''
        model = BertForMultipleChoice(config)


class TestBlip2Config(unittest.TestCase):
    """test blip2 config"""
    def clear_checkpoint_name_or_path(self, config):
        config.checkpoint_name_or_path = ''
        if config.vision_config is not None:
            config.vision_config.checkpoint_name_or_path = ''
        if config.text_config is not None:
            config.text_config.checkpoint_name_or_path = ''
        if config.qformer_config is not None:
            config.qformer_config.checkpoint_name_or_path = ''

    def test_init_model_for_qformer_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/blip2/run_blip2_stage1_vit_g_qformer_pretrain.yaml")
        self.clear_checkpoint_name_or_path(config)
        model = Blip2Qformer(config)

    def test_init_model_for_item_evaluator_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/blip2/run_blip2_stage1_vit_g_retrieval_flickr30k.yaml")
        self.clear_checkpoint_name_or_path(config)
        model = Blip2ItmEvaluator(config)

    def test_init_model_for_classifier_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/blip2/"
                                            "run_blip2_stage1_vit_g_zero_shot_image_classification_cifar100.yaml")
        self.clear_checkpoint_name_or_path(config)
        model = Blip2Classifier(config)

    def test_init_model_for_image_to_text_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/blip2/run_blip2_stage2_vit"
                                            "_g_baichuan_7b_image_to_text_generation.yaml")
        self.clear_checkpoint_name_or_path(config)
        model = Blip2ImageToTextGeneration(config)

    def test_init_model_for_llm_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/blip2/run_blip2_stage2_vit_g_baichuan_7b.yaml")
        self.clear_checkpoint_name_or_path(config)
        model = Blip2Llm(config)


class TestBloomConfig(unittest.TestCase):
    """test bloom config"""
    def test_init_model_for_text_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/bloom/run_bloom_560m.yaml")
        config.checkpoint_name_or_path = ''
        model = BloomLMHeadModel(config)


class TestClipConfig(unittest.TestCase):
    """test clip config"""
    def test_init_model_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/clip/run_clip_vit_b_16_pretrain_flickr8k.yaml")
        config.checkpoint_name_or_path = ''
        model = CLIPModel(config)


class TestCodegeex2Config(unittest.TestCase):
    """test Codegeex2 config"""
    def test_init_model_for_conditional_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/codegeex2/run_codegeex2_6b.yaml")
        config.checkpoint_name_or_path = ''
        model = ChatGLM2ForConditionalGeneration(config)


class TestCodeLlamaConfig(unittest.TestCase):
    """test codellama config"""
    def test_init_model_for_text_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/codellama/predict_codellama_34b_910b.yaml")
        config.checkpoint_name_or_path = ''
        model = LlamaForCausalLM(config)


class TestGLMConfig(unittest.TestCase):
    """test glm config"""
    def test_init_model_for_pretraining_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/glm/run_glm_6b_finetune.yaml")
        config.checkpoint_name_or_path = ''
        model = GLMForPreTraining(config)

    def test_init_model_for_text_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/glm/run_glm_6b_infer.yaml")
        config.checkpoint_name_or_path = ''
        model = GLMChatModel(config)


class TestGLM2Config(unittest.TestCase):
    """test glm2 config"""
    def test_init_model_for_conditional_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/glm2/run_glm2_6b.yaml")
        config.checkpoint_name_or_path = ''
        model = ChatGLM2ForConditionalGeneration(config)


class TestGPT2Config(unittest.TestCase):
    """test gpt2 config"""
    def test_init_model_for_text_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/gpt2/pretrain_gpt2_small_fp16.yaml")
        config.checkpoint_name_or_path = ''
        model = GPT2LMHeadModel(config)

    def test_init_model_for_sequence_classification_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/gpt2/finetune_gpt2_small_txtcls_fp16.yaml")
        config.checkpoint_name_or_path = ''
        model = GPT2ForSequenceClassification(config)


class TestLlamaConfig(unittest.TestCase):
    """test llama config"""
    def test_init_mode_for_text_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/llama/run_llama_7b.yaml")
        config.checkpoint_name_or_path = ''
        model = LlamaForCausalLM(config)


class TestLlama2Config(unittest.TestCase):
    """test llama2 config"""
    def test_init_model_for_text_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/llama2/run_llama2_7b.yaml")
        config.checkpoint_name_or_path = ''
        model = LlamaForCausalLM(config)


class TestMaeConfig(unittest.TestCase):
    """test mae config"""
    def test_init_model_for_pretraining_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/mae/run_mae_vit_base_p16_224_800ep.yaml")
        config.checkpoint_name_or_path = ''
        model = ViTMAEForPreTraining(config)


class TestPanguAlphaConfig(unittest.TestCase):
    """test PanguAlpha config"""
    def test_init_model_for_text_classification_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/pangualpha/run_pangualpha_2_6b_prompt_txtcls.yaml")
        config.checkpoint_name_or_path = ''
        model = PanguAlphaPromptTextClassificationModel(config)

    def test_init_model_for_text_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/pangualpha/run_pangualpha_2_6b.yaml")
        config.checkpoint_name_or_path = ''
        model = PanguAlphaHeadModel(config)


class TestSamConfig(unittest.TestCase):
    """test sam config"""
    def test_init_model_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/sam/run_sam_vit-b.yaml")
        config.checkpoint_name_or_path = ''
        model = SamModel(config)


class TestSwinConfig(unittest.TestCase):
    """test Swin config"""
    def test_init_model_for_image_classification_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/swin/run_swin_base_p4w7_224_100ep.yaml")
        config.checkpoint_name_or_path = ''
        model = SwinForImageClassification(config)


class TestT5Config(unittest.TestCase):
    """test T5 config"""
    def test_init_model_for_conditional_generation_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/t5/run_t5_tiny_on_wmt16.yaml")
        config.checkpoint_name_or_path = ''
        model = T5ForConditionalGeneration(config)


class TestVitConfig(unittest.TestCase):
    """test Vit config"""
    def test_init_model_for_image_classification_from_yaml(self):
        """test init model with config"""
        config = AutoConfig.from_pretrained("configs/vit/run_vit_base_p16_224_100ep.yaml")
        config.checkpoint_name_or_path = ''
        model = ViTForImageClassification(config)
