# 创建自定义数据集
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        # 声音可能存在多通道
        # signal-> (num_channels, samples) -> (2,16000) -> (1,16000)
        signal = self._resample_if_necessary(signal, sr)
        signal = signal.to(self.device)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal =signal[:, :self.num_samples] #取第1s
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)  # 两个数字分别表示左右填充0的个数
            # (1,1,2,1) 1,1表示在最后一个维度的左右填充，2，2表示在倒数第二个维度上填充
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # 聚合多个通道，并将其混合到一个通道
        if signal.shape[0] > 1:  # (2, 16000)
            signal = torch.mean(signal, dim=0, keepdim=True)  # 使用最小值的维度
        return signal


    def _get_audio_sample_path(self, index):
        # 获得数据的路径
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

if __name__ == "__main__":
    ANNOTATIONS_FILE = "../data/UrbanSound8K/metadata/UrbanSound8K.csv"
    AUDIO_DIR = "../data/UrbanSound8K/audio"
    SAMPLE_RATE = 16000  # 采样率
    NUM_SAMPLES = 22050  # 样本数量
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using devie {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )  # 将mel频谱图传递给usd
    # ms = mel_spectrogram(signal)
    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
