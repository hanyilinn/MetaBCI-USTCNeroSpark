import numpy as np
import struct
import torch
import torch.nn.functional as F
from scipy.signal import butter, lfilter
import time
import torch
import torch.nn as nn
from psychopy import visual, core, event, monitors
# import networks
class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        self.channel_num = input_size[1]
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[1]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[1] * 0.5), 1), (int(input_size[1] * 0.5), 1),
                                         int(self.pool*0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (2, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            #nn.Linear(15*22, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # print('输入的x的shape',x.shape)
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        out = out[:,:,0:self.channel_num:2,:]-out[:,:,1:self.channel_num:2,:]
        # if self.channel_num == 2:
        #     out = out.unsqueeze(2)
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)

        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        #out = out.view(out.size(0), -1)

        out = self.fc(out)
        return out

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def ReadCntFile(cntFileName="EEG.cnt"):
    data = []
    with open(cntFileName, "rb") as f:
        byte = f.read(4)
        num_channel = struct.unpack('i',byte)[0]
        byte = f.read(4)
        iNumEvtPerSample = struct.unpack('i',byte)[0]
        byte = f.read(4)
        iNumSplPerBlock = struct.unpack('i',byte)[0]
        byte = f.read(4)
        sampling_rate = struct.unpack('i',byte)[0]
        byte = f.read(4)
        iDataSize = struct.unpack('i',byte)[0]
        byte = f.read(4)
        resolution = struct.unpack('f',byte)[0]
        byte = f.read(int(num_channel*10))
        channel_names = struct.unpack('{}s'.format(int(num_channel*10)),byte)[0]
        channel_names = channel_names.decode('utf-8')
        while byte:
            try:
                # Do stuff with byte.
                byte = f.read(num_channel*iDataSize)
                data.append(struct.unpack(num_channel*'d', byte))
            except:
                break

        data = np.array(data)
    return data

def ReadCntFile_ly(cntFileName="EEG.cnt", numProcessSample=2048, isLatestSample=True, file_type='EEG'):
    data = []
    if file_type == 'EEG':
        channel_to_decode = 7
    elif file_type == 'PPG':
        channel_to_decode = 5
    else:
        raise Exception("Wrong data type! Please choose from 'EEG' and 'PPG'.")

    with open(cntFileName, "rb") as f:
        byte = f.read(4)
        num_channel = struct.unpack('i',byte)[0]
        byte = f.read(4)
        iNumEvtPerSample = struct.unpack('i',byte)[0]
        byte = f.read(4)
        iNumSplPerBlock = struct.unpack('i',byte)[0]
        byte = f.read(4)
        sampling_rate = struct.unpack('i',byte)[0]
        byte = f.read(4)
        iDataSize = struct.unpack('i',byte)[0]
        byte = f.read(4)
        resolution = struct.unpack('f',byte)[0]

        if isLatestSample:
            try:
                f.seek(-num_channel*(numProcessSample+1)*iDataSize, 2)
            except:
                data=ReadCntFile(cntFileName)
                # data = np.reshape(data,(-1,7))
                return data
            while byte:
                try:
                    # Do stuff with byte.
                    byte = f.read(channel_to_decode*iDataSize)
                    data.append(struct.unpack(channel_to_decode*'d', byte))
                except:
                    break
            data = np.array(data)
    return data

def Process_Online_data(num,device):
    data = ReadCntFile_ly('C:\CBCR\MuseData\Streaming\EEG.cnt', numProcessSample=num)
    if data.shape[0]>=num:
        EEG = data[-256:,1:3].transpose()
        EEG = butter_bandpass_filter(EEG, 0.3, 45, 256, order=6)
        EEG = torch.from_numpy(EEG, ).to(torch.float32).to(device)
        EEG = EEG.unsqueeze(0)
        return EEG
        
def main():
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
    print('已成功加载模型')
    
    while 1>0:
        try:
            EEG = Process_Online_data(num,device)
            output = model(EEG)
            probabilities = torch.sigmoid(output)
            value = probabilities[0][1].item()
            print("喜好度为：",value*100)
            # 更新文本框内容
            text_stim.text = f"喜好度为：{value*100:.2f}"
            # 清除屏幕并绘制新的文本框
            win.flip()
            text_stim.draw()
            # 刷新屏幕
            win.flip()
            ## 短暂等待
            if not event.getKeys():  # 如果没有按键事件，继续循环
                core.wait(0.1)
            
            # 检查是否按下退出键
            if 'escape' in event.getKeys() or 'q' in event.getKeys():
                break
        except:
            pass



if __name__ == "__main__":
    main()