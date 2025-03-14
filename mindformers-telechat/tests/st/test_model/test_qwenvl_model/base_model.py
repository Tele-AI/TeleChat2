# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""qwenvl Base Model."""
import os
import sys

MFPATH = os.path.abspath(__file__)
MFPATH = os.path.abspath(MFPATH + '/../../../../../')
sys.path.append(MFPATH)
sys.path.append(MFPATH + '/research/qwenvl')
# pylint: disable=C0413
from mindformers import MindFormerConfig

from research.qwenvl.qwenvl_model import QwenVL
from research.qwenvl.qwenvl_config import QwenVLConfig

VISION_MODEL_CONFIG = {
    'arch': {'type': 'QwenVLVisionModel'},
    'model_config': {
        'type': 'QwenVLVisionConfig',
        'num_hidden_layers': 2,
        'image_size': 448,
        'patch_size': 14
    },
}

LLM_MODEL_CONFIG = {
    'arch': {'type': 'QwenForCausalLM'},
    'model_config': {
        'type': 'QwenConfig',
        'num_layers': 2,
        'seq_length': 512,
        'vocab_size': 151936,
        'intermediate_size': 11008,
        'enable_slice_dp': False,
        'embedding_parallel_optimizer': False,
        'rms_norm_eps': 1.0e-6,
        'emb_dropout_prob': 0.0,
        'eos_token_id': 151643,
        'pad_token_id': 151643,
        'ignore_token_id': -100,
        'rotary_dtype': "float16",
        'use_flash_attention': True,
        'use_past': False,
        'is_dynamic': True,
        'num_blocks': 128,
        'top_k': 0,
        'top_p': 0.8,
        'do_sample': False,
        'enable_emb_opt': True,
        'rotary_pct': 1.0,
        'rotary_emb_base': 10000,
        'kv_channels': 128,
        'max_decode_length': 512
    }
}

BASE_CONFIG = {
    'trainer': {
        'type': 'MultiModalToTextGenerationTrainer',
        'model_name': 'qwenvl'
    },
    'train_dataset': {},
    'train_dataset_task': {},
    'micro_batch_interleave_num': 1,
    'parallel': {},
    'runner_config': {
        'epochs': 1,
        'batch_size': 4,
        'sink_mode': True,
        'sink_size': 2
    },
    'runner_wrapper': {
        'type': 'MFTrainOneStepCell',
        'scale_sense': {
            'type': 'DynamicLossScaleUpdateCell',
            'loss_scale_value': 64,
            'scale_factor': 2,
            'scale_window': 1000
        },
        'use_clip_grad': True,
    },
    'model': {
        'model_config': {
            'vision_model': VISION_MODEL_CONFIG,
            'llm_model': LLM_MODEL_CONFIG,
            'freeze_vision': True,
            'freeze_resampler': False,
            'freeze_llm': False,
            'use_past': False,
            'compute_dtype': "bfloat16",
            'param_init_type': "float16",
            'softmax_compute_type': "float32",
            'is_dynamic': True,
            'block_size': 32,
            'num_blocks': 128,
        }
    },
    'callbacks': [{'type': 'MFLossMonitor'}]
}

def get_train_config():
    return MindFormerConfig(**BASE_CONFIG)

def get_predict_config():
    config = MindFormerConfig(**BASE_CONFIG)
    config.model.model_config.compute_dtype = 'float16'
    config.model.model_config.use_past = True
    config.model.model_config.block_size = 16
    config.model.model_config.num_blocks = 512
    config.model.model_config.llm_model.model_config.use_past = True
    config.model.model_config.llm_model.model_config.block_size = 16
    config.model.model_config.llm_model.model_config.num_blocks = 512
    return config

def get_model_config(config):
    """get instanced model config."""
    model_config = config.model.model_config
    return QwenVLConfig(**model_config)

def get_model(config):
    """get instanced model."""
    return QwenVL(config)
