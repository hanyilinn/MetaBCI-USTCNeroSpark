import numpy as np
import struct
import torch
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter
import time
import re
import os
import torch
import torch.nn as nn

from datetime import datetime
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

# 定义一个函数来转换时间字符串为时间戳
def convert_to_timestamp(time_str):
    # 清理字符串，移除干扰字符
    cleaned_str = re.sub(r'^[^0-9]+(.*)', r'\1', time_str)
    print(cleaned_str)
    return round(1000*(datetime.strptime(cleaned_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()))

# 定义一个函数来解析scores.txt文本文件
def parse_txt_file(file_path):
    stats = []
    with open(file_path, 'r', encoding='gbk') as file:
        lines = file.readlines()
        segment_start_idx = 0  # 片段起始索引

        # 按片段读取，每个片段有6行数据
        for i in range(0, len(lines), 7):  # 步长为7，确保每次处理7行
            try:
                # 读取每个片段的第二行和第三行
                stat_time_str = lines[i + 1].strip()
                end_time_str = lines[i + 2].strip()

                # 转换为时间戳
                stat_timestamp = convert_to_timestamp(stat_time_str)
                end_timestamp = convert_to_timestamp(end_time_str)

                # 读取liking, valence, arousen的值
                liking_str = lines[i + 3].split(': ')[1].replace("\n", "")
                valence_str = lines[i + 4].split(': ')[1].replace("\n", "")
                arousen_str = lines[i + 5].split(': ')[1].replace("\n", "")

                liking = np.int64(float(liking_str))
                valence = np.int64(float(valence_str))
                arousen = np.int64(float(arousen_str))

                # 存储结果
                stats.append({
                    'start_time': stat_timestamp,
                    'end_time': end_timestamp,
                    'liking': liking,
                    'valence': valence,
                    'arousen': arousen
                })
            except IndexError:
                # 如果索引超出范围，停止读取
                break
            except ValueError:
                # 如果转换数值失败，打印错误并继续
                print(f"Error converting values to float at line {i + 3} to {i + 5}")

    return stats


def main():
    step = 128
    window_length = 256
    data_for_exper = []
    datapath = [1722078786118,1722497227125,1722510612662,1722592557872]
    i = 2
    for data_path in datapath:
        data = ReadCntFile('C:\CBCR\MuseData\\'+str(data_path)+'\EEG.cnt')
        data[:, 5] = data[:, 5].astype(np.int64)
        data_label = parse_txt_file('D:\MetaBCI-master\scores'+str(i)+'.txt')
        for label in data_label:
            if label['valence'] in [0,1,2,3]:
                flag = 0
            elif label['valence'] in [6,7,8,9]:
                flag = 1
            else:
                break
            # 根据 start_time 和 end_time 截取 data 中的时间戳对应的数据
            print('视频开始时间为：',label['start_time'])
            print('视频结束时间为：',label['end_time'])
            start_idx = np.searchsorted(data[:, 5], label['start_time'], side='left')
            end_idx = np.searchsorted(data[:, 5], label['end_time'], side='right')
            # 截取 data 中的时间戳为 start_time 和 end_time 之间的数据
            selected_data = data[start_idx:end_idx, 1:3].T
            EEG = butter_bandpass_filter(selected_data, 0.3, 45, 256, order=6)
            for start in range(0, EEG.shape[1] - window_length + 1, step):
                window = EEG[:, start:start + window_length]
                
                data_for_exper.append((window, flag))  # 假设标签 'a' 对应于所有窗口
        i += 1
        
    X = np.array([item[0] for item in data_for_exper]) # 数据
    print(X.shape)
    y = np.array([item[1] for item in data_for_exper]) # 标签
    print(y.shape)
    #深度学习，启动！    
    data_train, data_test, targets_train, targets_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 将数据和标签转换为torch张量
    data_train = torch.tensor(data_train).to(torch.float32)
    data_test = torch.tensor(data_test).to(torch.float32)
    targets_train = torch.tensor(targets_train).long()
    targets_test = torch.tensor(targets_test).long()

    # 创建数据加载器
    train_dataset = TensorDataset(data_train, targets_train)
    test_dataset = TensorDataset(data_test, targets_test)

    batch_size = 64  # 根据你的硬件设置适当的批量大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型，设置超参数
    num_classes = 2  # 假设targets是numpy数组，需要计算类别数
    model = TSception(num_classes=num_classes, input_size=(1,2,256), sampling_rate=256, num_T=15, num_S=15,
                    hidden=32, dropout_rate=0.5)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    num_epochs = 50  # 设置训练的轮数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # print('inputs的shape',inputs.shape)
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        torch.save(model.state_dict(), str(epoch)+'model.pth')
        # 在测试集上验证模型
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        accuracy = correct / len(test_loader.dataset)
        print(f"Accuracy of the model on the test images: {accuracy * 100}%")

# 保存训练好的模型
# 

"""
input = torch.from_numpy(EEG[np.newaxis,np.newaxis,:,:])
output = model(input)
softmax_out = F.softmax(output, dim=1).detach().numpy()
score = softmax_out[0, 1] * 100
score = format(score, '.3f')
"""
                # print(score)





if __name__ == "__main__":
    main()