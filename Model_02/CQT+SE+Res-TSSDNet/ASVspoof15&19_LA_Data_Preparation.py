import soundfile as sf
import pandas as pd
import numpy as np
import torch
from scipy import signal
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import librosa
from spafe.features.cqcc import cqt_spectrogram
from spafe.features.cqcc import cqcc
from spafe.features.lfcc import lfcc
from spafe.utils.preprocessing import SlidingWindow


"""
This program generates 
    1) equal-duration time domain raw waveforms
    2) 2D log power of constant Q transform 
from ASVspoof2019 and ASVspoof2015 official datasets, respectively. 
This program is supposed to be run independently for data preparation.

Official dataset download link: https://www.asvspoof.org/database

The CQT parameter settings follow the ones used in:
X. Li et al., "Replay and synthetic speech detection with res2net architecture," in Proc. ICASSP 2021.

"""


def gen_time_frame_19(protocol_path, read_audio_path, write_audio_path, duration, status: str):
    sub_path = write_audio_path + status + '_' + str(duration) + '/'
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path, sep=' ', header=None).values
    file_index = protocol[:, 1]
    num_files = protocol.shape[0]
    total_sample_count = 0

    for i in range(num_files):
        x, fs = sf.read(read_audio_path + file_index[i] + '.flac')
        if len(x) < duration * fs:
            x = np.tile(x, int((duration * fs) // len(x)) + 1)
        x = x[0: (int(duration * fs))]
        total_sample_count += 1
        sf.write(sub_path + file_index[i] + '.flac', x, fs)
    print('{} pieces {}-second {} samples generated.'.format(total_sample_count, duration, status))


def gen_cqt_19(protocol_path, read_audio_path, write_audio_path, duration=6.4, status='train'):
    sub_path = write_audio_path + status + '_' + str(duration) + '_cqt' + '/'
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path, sep=' ', header=None).values
    file_index = protocol[:, 1]
    num_files = protocol.shape[0]
    total_sample_count = 0
    fs = 16000

    for i in range(num_files):
        x, fs = sf.read(read_audio_path + file_index[i] + '.flac')
        len_sample = int(duration * fs)

        if len(x) < len_sample:
            x = np.tile(x, int(len_sample // len(x)) + 1)

        x = x[0: int(len_sample - 256)]

        x = signal.lfilter([1, -0.97], [1], x)
        # x_cqt = librosa.cqt(x, sr=fs, hop_length=256, n_bins=432, bins_per_octave=48, window='hann', fmin=15)
        x_cqt = cqt_spectrogram(x,
                                fs=fs,
                                pre_emph=0, # 0为false，运用预加重
                                pre_emph_coeff=0.97,  # 预加重滤波系数
                                window=SlidingWindow(0.03, 0.015, "hamming"),
                                #论文中数据
                                #window=SlidingWindow(0.016, 0.015, "hamming"),
                                nfft=2048,  # FFT点的个数
                                low_freq=0, # mel滤波器的最低带边缘(Hz)。(默认为0)
                                high_freq=fs/2)  # mel滤波器的最高频带边缘(Hz)。(默认是采样/ 2)
        pow_cqt = np.square(np.abs(x_cqt))
        log_pow_cqt = 10 * np.log10(pow_cqt + 1e-30)
        total_sample_count += 1
        torch.save(log_pow_cqt, sub_path + file_index[i] + '.pt')
    print('{} {} CQT features of {}*{} generated.'.format(total_sample_count, status, 432, int((duration*fs)//256)))


def gen_cqcc_19(protocol_path, read_audio_path, write_audio_path, duration=6.4, status='train'):
        sub_path = write_audio_path + status + '_' + str(duration) + '_cqcc' + '/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        protocol = pd.read_csv(protocol_path, sep=' ', header=None).values
        file_index = protocol[:, 1]
        num_files = protocol.shape[0]
        total_sample_count = 0
        fs = 16000

        for i in range(num_files):
            x, fs = sf.read(read_audio_path + file_index[i] + '.flac')
            len_sample = int(duration * fs)
            # 处理音频文件（在干嘛不知道）
            if len(x) < len_sample:
                x = np.tile(x, int(len_sample // len(x)) + 1)

            x = x[0: int(len_sample - 256)]

            x = signal.lfilter([1, -0.97], [1], x)

            # x_cqt = librosa.cqt(x, sr=fs, hop_length=256, n_bins=432, bins_per_octave=48, window='hann', fmin=15)

            x_cqcc = cqcc(x,
                          fs=fs,
                          pre_emph=1, #1为true，运用预加重
                          num_ceps = 20,
                          pre_emph_coeff=0.97, #预加重滤波系数
                          window=SlidingWindow(0.03, 0.015, "hamming"),
                          nfft=2048,  # FFT点的个数
                          low_freq=0,  # mel滤波器的最低带边缘(Hz)。(默认为0)
                          high_freq=fs/2, # mel滤波器的最高频带边缘(Hz)。(默认是采样/ 2)
                          normalize="mvn")

            pow_cqcc = np.square(np.abs(x_cqcc))
            log_pow_cqcc = 10 * np.log10(pow_cqcc + 1e-30)
            total_sample_count += 1
            torch.save(log_pow_cqcc, sub_path + file_index[i] + '.pt')

        print('{} {} CQCC features of {}*{} generated.'.format(total_sample_count, status, 432, int((duration * fs) // 256)))


def gen_lfcc_19(protocol_path, read_audio_path, write_audio_path, duration=6.4, status='train'):
    sub_path = write_audio_path + status + '_' + str(duration) + '_lfcc' + '/'
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)

    protocol = pd.read_csv(protocol_path, sep=' ', header=None).values
    file_index = protocol[:, 1]
    num_files = protocol.shape[0]
    total_sample_count = 0
    fs = 16000

    for i in range(144131,num_files):
        x, fs = sf.read(read_audio_path + file_index[i] + '.flac')
        len_sample = int(duration * fs)
        # 处理音频文件（在干嘛不知道）
        if len(x) < len_sample:
            x = np.tile(x, int(len_sample // len(x)) + 1)

        x = x[0: int(len_sample - 256)]

        x = signal.lfilter([1, -0.97], [1], x)

        x_lfcc = lfcc(x,  #用于计算特征的单声道音频信号(Nx1)
                      fs=fs, #我们正在处理的信号的采样频率。(默认为16000)
                      pre_emph=1, #1为true，运用预加重
                      pre_emph_coeff=0.97, #预加重滤波系数
                      window=SlidingWindow(0.03, 0.015, "hamming"),
                      nfilts=20, #滤波器组中的滤波器数量
                      nfft=512, #FFT点的个数
                      low_freq=0, #mel滤波器的最低带边缘(Hz)。(默认为0)
                      high_freq=8000, #mel滤波器的最高频带边缘(Hz)。(默认是采样/ 2)
                      normalize="mvn") #规范化

        pow_lfcc = np.square(np.abs(x_lfcc))
        log_pow_lfcc = 10 * np.log10(pow_lfcc + 1e-30)
        total_sample_count += 1
        torch.save(log_pow_lfcc, sub_path + file_index[i] + '.pt')

    print('{} {} LFCC features of {}*{} generated.'.format(total_sample_count, status, 432, int((duration * fs) // 256)))


if __name__ == '__main__':
    # TODO: ASVspoof2019 data preparation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # directory info of ASVspoof2019 dataset
    root_path = '../LAPA/'
    train_protocol_path = root_path + 'ASVspoof2019_LAPA_cm_protocols/ASVspoof2019.LAPA.cm.train.trn.txt'
    dev_protocol_path   = root_path + 'ASVspoof2019_LAPA_cm_protocols/ASVspoof2019.LAPA.cm.dev.trl.txt'
    eval_protocol_path  = root_path + 'ASVspoof2019_LAPA_cm_protocols/ASVspoof2019.LAPA.cm.eval.trl.txt'
    train_data_path     = root_path + 'ASVspoof2019_LAPA_train/flac/'
    dev_data_path       = root_path + 'ASVspoof2019_LAPA_dev/flac/'
    eval_data_path      = root_path + 'ASVspoof2019_LAPA_eval/flac/'

    # create folders for new types of data
    new_data_path = root_path + 'data/'
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)

    # generate equal-duration time-frames examples
    # print('Generating time-frame data...')
    # time_dur = 6  # in seconds
    # gen_time_frame_19(train_protocol_path, train_data_path, new_data_path, duration=time_dur, status='train')
    # gen_time_frame_19(dev_protocol_path,   dev_data_path,   new_data_path, duration=time_dur, status='dev')
    # gen_time_frame_19(eval_protocol_path,  eval_data_path,  new_data_path, duration=time_dur, status='eval')

    # generate cqt feature per sample
    print('Generating CQT data...')
    cqt_dur = 6.4  # in seconds, default ICASSP 2021 setting
    gen_cqt_19(train_protocol_path, train_data_path, new_data_path, duration=cqt_dur, status='train')
    gen_cqt_19(dev_protocol_path,   dev_data_path,   new_data_path, duration=cqt_dur, status='dev')
    gen_cqt_19(eval_protocol_path,  eval_data_path,  new_data_path, duration=cqt_dur, status='eval')

    # generate cqcc feature per sample
    print('Generating CQCC data...')
    # cqcc_dur = 6.4
    # gen_cqcc_19(train_protocol_path, train_data_path, new_data_path, duration=cqcc_dur, status='train')
    # gen_cqcc_19(dev_protocol_path,   dev_data_path,   new_data_path, duration=cqcc_dur, status='dev')
    # gen_cqcc_19(eval_protocol_path,  eval_data_path,  new_data_path, duration=cqcc_dur, status='eval')

    # generate lfcc feature per sample
    # print('Generating LFCC data...')
    # lfcc_dur = 6.4
    # gen_lfcc_19(train_protocol_path, train_data_path, new_data_path, duration=lfcc_dur, status='train')
    # gen_lfcc_19(dev_protocol_path,   dev_data_path,   new_data_path, duration=lfcc_dur, status='dev')
    # gen_lfcc_19(eval_protocol_path,  eval_data_path,  new_data_path, duration=lfcc_dur, status='eval')

    print('End of Program.')
