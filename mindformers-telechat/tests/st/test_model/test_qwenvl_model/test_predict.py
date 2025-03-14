# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Test module for testing the qwenvl interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_qwenvl_model/test_predict.py
"""
import os
import sys
import pytest
import numpy as np

import mindspore as ms

MFPATH = os.path.abspath(__file__)
MFPATH = os.path.abspath(MFPATH + '/../../../../../')
sys.path.append(MFPATH)
sys.path.append(MFPATH + '/research/qwenvl')

# pylint: disable=C0413
from mindformers import MindFormerRegister, MindFormerModuleType

from research.qwenvl.qwenvl_config import QwenVLConfig, QwenConfig
from research.qwenvl.qwenvl_model import QwenVL
from research.qwenvl.qwenvl_processor import QwenVLImageProcessor, QwenVLProcessor
from research.qwenvl.qwenvl_tokenizer import QwenVLTokenizer
from research.qwenvl.qwenvl_transform import QwenVLTransform
from research.qwenvl.qwen_model import QwenForCausalLM

from .base_model import get_predict_config, get_model, get_model_config

ms.set_context(mode=0, jit_config={'jit_level': 'O0', 'infer_boost': 'on'})


def register_modules():
    """register modules"""
    MindFormerRegister.register_cls(QwenVL, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(QwenForCausalLM, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(QwenConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLTokenizer, MindFormerModuleType.TOKENIZER)
    MindFormerRegister.register_cls(QwenVLTransform, MindFormerModuleType.TRANSFORMS)
    MindFormerRegister.register_cls(QwenVLProcessor, MindFormerModuleType.PROCESSOR)
    MindFormerRegister.register_cls(QwenVLImageProcessor, MindFormerModuleType.PROCESSOR)


def generate_coord(img_start_pos):
    num_img = len(img_start_pos)
    coord = np.zeros((num_img, 256, 2), np.int32)
    for idx, pos in enumerate(img_start_pos):
        for img_pos in range(256):
            coord[idx, img_pos] = [0, pos + img_pos]
    return coord


class TestQwenVLPredict:
    """A test class for testing model prediction."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_predict(self):
        """
        Feature: model.generate()
        Description: Test model for predict.
        Expectation: TypeError, ValueError, RuntimeError
        """
        register_modules()
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        config = get_predict_config()
        model_config = get_model_config(config)
        model = get_model(model_config)
        input_ids = np.random.randint(0, 128, size=(1, 256), dtype=np.int32)
        images = ms.Tensor(np.random.random(size=(1, 1, 3, 448, 448)), dtype=ms.float32)
        img_pos = ms.Tensor([generate_coord([30])], ms.int32)
        _ = model.generate(input_ids=input_ids,
                           images=images,
                           img_pos=img_pos)
