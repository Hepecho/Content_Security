import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from UrbanSoundDataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
Epochs = 10
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = "../data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "../data/UrbanSound8K/audio"
SAMPLE_RATE = 22050  # 采样率
NUM_SAMPLES = 22050  # 样本数量


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, targets in data_loader:
        input, targets = input.to(device), targets.to(device)

        # calculate loss #每一个batch计算loss
        # 使用当前模型获得预测
        predictions = model(input)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimiser.zero_grad()  # 在每个batch中让梯度重新为0
        loss.backward()  # 反向传播
        optimiser.step()

    print(f"Loss: {loss.item()}")  # 打印最后的batch的loss


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------")
    print("Training is down.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")
    # instantiating our dataset object and create data loader
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

    train_dataloader = create_data_loader(usd, BATCH_SIZE)
    # build model

    cnn = CNNNetwork().to(device)
    print(cnn)

    # instantiate loss function + opptimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, Epochs)

    torch.save(cnn.state_dict(), "../model/feedforwardnet.pth")
    print("Model trained and stored at ../model/feedforwardnet.pth")
