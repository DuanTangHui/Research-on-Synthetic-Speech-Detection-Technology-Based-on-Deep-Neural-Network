import models
import soundfile as sf
import numpy as np
import torch
from scipy import signal
from spafe.features.cqcc import cqt_spectrogram
from spafe.utils.preprocessing import SlidingWindow
import torch.nn.functional as F
from flask import Flask, request, make_response, jsonify


app = Flask(__name__)

classes = ["Real Aduio","Fake Audio"]
duration = 6
fs = 1600

def asv_cal_accuracies(path_data,net,data_type='time_frame', dataset=19):
    net.eval()
    with torch.no_grad():
        if data_type == 'time_frame':
            sample, _ = sf.read(path_data)
            sample = torch.tensor(sample, dtype=torch.float32)
            # 扩充sample的维度为1x1x16000
            sample = torch.unsqueeze(sample, 0)
            sample = torch.unsqueeze(sample, 1)
        if data_type == 'CQT':
            sample = torch.load(path_data)
            sample = torch.tensor(sample, dtype=torch.float32)
            # 扩充sample的维度为1x1x16000
            sample = torch.unsqueeze(sample, 0)
            sample = torch.unsqueeze(sample, 1)
            # sample = torch.unsqueeze(sample, 0)

        preds = net(sample)
        t = F.softmax(preds, dim=1)
    return t


@app.route('/')
def index():
    return "hello"

@app.route('/special', methods=['POST'])
def classify1():
    # 获取待检测音频数量
    length = request.form.get('number')
    # 存储检测结果
    result = []

    for i in range(int(length)):
        # 依次获取文件数据
        raw = 'files[' + str(i) + ']'
        name = 'filename[' + str(i) + ']'
        audio = request.files.get(raw)
        name = request.form.get(name)
        data_path = "./audio/special_audio.flac"

        # 预处理音频文件 截断或重复至6s
        x, fs = sf.read(audio)
        if len(x) < duration * fs:
            x = np.tile(x, int((duration * fs) // len(x)) + 1)
        x = x[0: (int(duration * fs))]
        sf.write(data_path, x, fs)

        # 送入模型训练
        Net = models.DilatedNet()
        check_point = torch.load('./model/time_frame_48_ASVspoof2019_LA_Loss_0.0009_dEER_0.82%_eEER_4.4%.pth',
                                 map_location='cpu')
        Net.load_state_dict(check_point['model_state_dict'])

        # 返回预测结果 【真的概率，假的概率】
        preds = asv_cal_accuracies(data_path, Net, data_type='time_frame', dataset=19)

        # 获取较大的值为预测结果
        i = preds.argmax(dim=1) # i=0为真 ，1 为假
        # label = classes[i]
        # 将结果填入result
        result.append({
            'nameOf': name,
            'state': int(i),
            'confidence': float(preds[0][i].item() * 100)
        })
    # result = "{}: {:.10f}%".format(label, preds[0][i].item() * 100)
    response = make_response(jsonify(result))
    response.headers["Access-Control-Allow-Origin"] = '*'
    response.headers["Access-Control-Allow-Methods"] = 'POST'
    response.headers["Access-Control-Allow-Headers"] = "x-requested-with,content-type"
    return response

@app.route('/general', methods=['POST'])
def classify2():
    # 获取待检测音频数量
    length = request.form.get('number')
    # 存储检测结果
    result = []

    for i in range(int(length)):
        # 依次获取文件数据
        raw = 'files[' + str(i) + ']'
        name = 'filename[' + str(i) + ']'
        audio = request.files.get(raw)
        name = request.form.get(name)

        data_path = "./audio/general_audio.pt"

        # 预处理音频文件
        x, fs = sf.read(audio)
        len_sample = int(duration * fs)
        if len(x) < len_sample:
            x = np.tile(x, int(len_sample // len(x)) + 1)
        x = x[0: int(len_sample - 256)]
        x = signal.lfilter([1, -0.97], [1], x)
        x_cqt = cqt_spectrogram(x,
                                fs=fs,
                                pre_emph=0,  # 0为false，运用预加重
                                pre_emph_coeff=0.97,  # 预加重滤波系数
                                window=SlidingWindow(0.03, 0.015, "hamming"),
                                # 论文中数据
                                # window=SlidingWindow(0.016, 0.015, "hamming"),
                                nfft=2048,  # FFT点的个数
                                low_freq=0,  # mel滤波器的最低带边缘(Hz)。(默认为0)
                                high_freq=fs / 2)  # mel滤波器的最高频带边缘(Hz)。(默认是采样/ 2)
        pow_cqt = np.square(np.abs(x_cqt))
        log_pow_cqt = 10 * np.log10(pow_cqt + 1e-30)
        torch.save(log_pow_cqt, data_path)
        # 送入模型训练
        Net = models.SSDNet2D()
        check_point = torch.load('./model/CQT_20_ASVspoof2019_LAPA_Loss_0.017_dEER_1.75%_eEER_5.07%.pth',
                                 map_location='cpu')
        Net.load_state_dict(check_point['model_state_dict'])
        # 返回预测结果 【真的概率，假的概率】
        preds = asv_cal_accuracies(data_path, Net, data_type='CQT', dataset=19)
        # 获取较大的值为预测结果
        i = preds.argmax(dim=1)  # i=0为真 ，1 为假
        # label = classes[i]
        # 将结果填入result
        result.append({
            'nameOf': name,
            'state': int(i),
            'confidence': float(preds[0][i].item() * 100)
        })
    response = make_response(jsonify(result))
    response.headers["Access-Control-Allow-Origin"] = '*'
    response.headers["Access-Control-Allow-Methods"] = 'POST'
    response.headers["Access-Control-Allow-Headers"] = "x-requested-with,content-type"
    return response


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True, host='0.0.0.0',port=8050)