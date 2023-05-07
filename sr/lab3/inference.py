import torch
import torchaudio
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report

from cnn import CNNNetwork
from UrbanSoundDataset import UrbanSoundDataset
from train import ANNOTATIONS_FILE, AUDIO_DIR, NUM_SAMPLES, SAMPLE_RATE

class_mapping =[
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) ->[[0.1, 0.01, ... ,0.6]] #概率最大的即为所选
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("../model/feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset
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
                            "cpu")

    # get a sample from urban sound dataset for inference
    print(usd.__len__())
    true_list = []
    pred_list = []
    for data in usd:
        input, target = data[0], data[1]  # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)
        # make an inference
        predicted, expected = predict(cnn, input, target, class_mapping)
        pred_list.append(predicted)
        true_list.append(expected)

    cm = confusion_matrix(true_list, pred_list, labels=class_mapping)
    print(cm.shape)
    # print('cm is:\n', cm)
    # print(f"Predicted:'{predicted}', expected:'{expected}'")
    conf_matrix = pd.DataFrame(cm, index=class_mapping, columns=class_mapping)  # 数据有5个类别
    # 画出混淆矩阵
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 10}, cmap="Blues")
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('confusion.pdf', bbox_inches='tight')
    plt.show()

    # 2.计算accuracy
    print('accuracy_score', accuracy_score(true_list, pred_list))

    # 3.计算多分类的precision、recall、f1-score分数
    print('Micro precision', precision_score(true_list, pred_list, average='micro'))
    print('Micro recall', recall_score(true_list, pred_list, average='micro'))
    print('Micro f1-score', f1_score(true_list, pred_list, average='micro'))

    print('Macro precision', precision_score(true_list, pred_list, average='macro'))
    print('Macro recall', recall_score(true_list, pred_list, average='macro'))
    print('Macro f1-score', f1_score(true_list, pred_list, average='macro'))

    # 下面这个可以显示出每个类别的precision、recall、f1-score。
    print('classification_report\n', classification_report(true_list, pred_list))

