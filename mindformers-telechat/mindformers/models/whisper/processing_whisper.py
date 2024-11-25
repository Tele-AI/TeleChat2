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
Whisper FeatureExtractor
"""
import numpy as np
import soundfile as sf
import librosa

from mindformers.dataset.transforms.audio_utils import spectrogram, window_function, mel_filter_bank


class WhisperFeatureExtractor:
    """FeatureExtractor for processing audio data"""
    def __init__(self, feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400,
                 padding_value=0.0):
        self.feature_size = feature_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def __call__(self, audio_path, **kwargs):
        with open(audio_path, "rb") as file:
            array, sampling_rate = sf.read(file)
            array = array.T
        # resample
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        # pad
        array = self.pad(array, max_length=self.n_samples)
        # compute the log-mel spectrogram
        input_features = self._np_extract_fbank_features(array)
        return input_features

    def pad(self, array, max_length, truncation=True):
        length = array.shape[0]
        if length < max_length:
            array = np.pad(array, (0, self.n_samples - length), constant_values=(0, self.padding_value))
        elif length > max_length and truncation:
            array = array[:max_length]
        return array

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters,
            log_mel="log10",
        )
        log_spec = log_spec[:, :-1]
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
