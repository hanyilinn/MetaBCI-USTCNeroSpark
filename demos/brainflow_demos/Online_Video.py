import numpy as np
import struct
import torch
import torch.nn.functional as F
from scipy.signal import butter, lfilter
import torch
import torch.nn as nn
from psychopy import visual, core, event, monitors
import sys
sys.path.append('E:\metametametabci\metabci')
from brainda.algorithms.deep_learning.Tception import TSception
#from brainda.paradigms import Video
from brainflow.amplifiers import MuseScan
from brainda.datasets.MuseData import MuseData
def main():
    dataset = MuseData()
    process = MuseScan()
    mon = monitors.Monitor("primary_monitor", ...)
    # 创建窗口
    win = visual.Window(
        size=(400, 100),  # 分辨率
        pos = (0,0),
        screen=0,  # 屏幕编号
        units="pix",  # 单位
        fullscr=False,  # 全屏
        allowGUI=False,  # 不允许GUI
        autoLog=False,  # 不记录日志
        monitor=mon  # 使用上面定义的监视器
    )
    # 创建文本显示框
    text_stim = visual.TextStim(win, text="预测的liking值是: 0%", pos=(0, 0), color='red', height=20)
    num=256
    model = TSception(num_classes=2, input_size=(1,2,256), sampling_rate=256, num_T=15, num_S=15,hidden=32, dropout_rate=0.5)
    model_path = 'D:\MetaBCI-master\demos\\brainflow_demos\\49model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    while 1>0:
        try:
            EEG = process.recv(num)
            EEG = dataset.butter_bandpass_filter(EEG, 0.3, 45, 256, order=6)
            EEG = torch.from_numpy(EEG, ).to(torch.float32).to(device)
            EEG = EEG.unsqueeze(0)
            output = model(EEG)
            probabilities = torch.sigmoid(output)
            value = probabilities[0][1].item()
            print("喜好度为：",value*100)
            text_stim.text = f"喜好度为：{value*100:.2f}"
            win.flip()
            text_stim.draw()
            win.flip()
            if not event.getKeys():  
                core.wait(0.1)
            if 'escape' in event.getKeys() or 'q' in event.getKeys():
                break
        except:
            pass

if __name__ == "__main__":
    main()
