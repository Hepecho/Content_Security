import wave
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display


def read_wav(path_wav):
    f = wave.open(path_wav, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]  # 通道数、采样字节数、采样率、采样帧数
    voiceStrData = f.readframes(nframes)
    waveData = np.frombuffer(voiceStrData, dtype=np.short)  # 将原始字符数据转换为整数
    waveData = waveData * 1.0 / max(abs(waveData))  # 音频数据归一化, instead of .fromstring
    waveData = np.reshape(waveData, [nframes, nchannels]).T
    # .T 表示转置, 将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共nchannels行
    f.close()
    return waveData, nframes, framerate


def draw_time_domain_image(waveData, nframes, framerate):       # 时域特征
    time = np.arange(0,nframes) * (1.0/framerate)
    plt.plot(time,waveData[0,:],c='b')
    plt.xlabel('time')
    plt.ylabel('am')
    plt.show()


def draw_frequency_domain_image(waveData):      # 频域特征
    fftdata = np.fft.fft(waveData[0, :])
    fftdata = abs(fftdata)
    hz_axis = np.arange(0, len(fftdata))
    plt.figure()
    plt.plot(hz_axis, fftdata, c='b')
    plt.xlabel('hz')
    plt.ylabel('am')
    plt.show()


def draw_Spectrogram(waveData, framerate):
    framelength = 0.025  # 帧长20~30ms
    framesize = framelength * framerate
    # 每帧点数 N = t*fs,通常情况下值为256或512,要与NFFT相等, 而NFFT最好取2的整数次方,即framesize最好取的整数次方
    nfftdict = {}
    lists = [32, 64, 128, 256, 512, 1024]
    for i in lists:  # 找到与当前framesize最接近的2的正整数次方
        nfftdict[i] = abs(framesize - i)
    sortlist = sorted(nfftdict.items(), key=lambda x: x[1])  # 按与当前framesize差值升序排列
    framesize = int(sortlist[0][0])  # 取最接近当前framesize的那个2的正整数次方值为新的framesize
    NFFT = framesize  # NFFT必须与时域的点数framsize相等，即不补零的FFT
    overlapSize = 1.0 / 3 * framesize  # 重叠部分采样点数overlapSize约为每帧点数的1/3~1/2
    overlapSize = int(round(overlapSize))  # 取整
    spectrum, freqs, ts, fig = plt.specgram(waveData[0], NFFT=NFFT, Fs=framerate, window=np.hanning(M=framesize),
                                            noverlap=overlapSize, mode='default', scale_by_freq=True, sides='default',
                                            scale='dB', xextent=None)  # 绘制Spectrogram图
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.title('Spectrogram')
    plt.show()

def mfcc_librosa(path):
    y, sr = librosa.load(path, sr=None)
    mfcc_data = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(mfcc_data.shape)
    # plt.matshow(mfcc_data)
    # plt.title('MFCC')
    # plt.show()
    librosa.display.specshow(mfcc_data, sr=sr, x_axis='time')
    plt.show()


if __name__ == '__main__':

    path_wav = './data/TAL_SER/wav/dev/S0005/102071.wav'
    waveData, nframes, framerate = read_wav(path_wav)
    draw_Spectrogram(waveData, framerate)
    mfcc_librosa(path_wav)
