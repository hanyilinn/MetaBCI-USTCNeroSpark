# 基于脑电信号的短视频喜好度评估系统

#### USTC-NeroSpark团队

## 概要

本系统主要由**3个模块**组成：

1）视频刺激诱发模块

2）离线训练模块

3）在线测试模块

本文档将分别予以详细介绍。

### 一、实验设备与实验设置

本研究使用消费级Muse头环采集参与者的EEG数据。Muse头环具备4个EEG通道，根据国际10-20系统，这些通道分别对应于额叶AF7和AF8以及颞叶TP9和TP10,前额中心Fpz作为参考信号。所有电极均为干式电极，方便用户佩戴。脑电信号采样率为256Hz。考虑到信号采集质量，本研究只使用AF7和AF8两个通道进行后续处理与分析，采集到的EEG信号通过蓝牙传输到计算机设备。

受试者为一名23岁成年男性，共进行4次实验，每次实验时间控制在45分钟左右。受试者持有本科学位，具备正常的裸眼视力，且无神经或精神疾病以及头部创伤史，为右手使用习惯者。实验前，已被充分告知实验目的，完全了解实验程序，并已签署知情同意书。

实验所需视频来源于抖音软件，共采集135个时长在30秒左右的短视频，为减少软件推荐算法产生的影响，135个短视频分别由5个账号收集完成，每次实验将随机播放未播放的视频。

### 二、视频刺激诱发模块

与Meta BCI已提供的P300、SSVEP等功能类似，我们开发的视频刺激诱发模块封装在 `brainstim`文件夹中 `paradigm.py`的 `mp4play`块，并在 `stim_demo.py`中进行调用。

#### mp4play：继承自 `VisualStim`，提供专用于处理MP4视频播放和用户反馈的类。

```python
def __init__(self, win, colorSpace="rgb", allowGUI=True):
```

- **参数**:
  - `win`: 视频将播放的窗口。
  - `colorSpace` (str): 视频的颜色空间，默认为"rgb"。
  - `allowGUI` (bool): 是否允许使用GUI组件，默认为True。
- **说明**: 初始化 `mp4play`实例，设置视频文件夹路径并加载可用的MP4文件。

##### 1.mp4gain

```python
def mp4gain(self):
```

- **说明**: 随机选择一个尚未播放的MP4视频文件并返回其路径。如果所有视频都已播放，则通知用户。

##### 2.feedback

```python
def feedback(self, video_path):
```

- **参数**:
  - `video_path` (str): 要播放的视频的文件路径。
- **说明**: 使用VLC播放视频，并处理用户反馈进行评分。将分数保存到文本文件中。

##### 3.get_scores

```python
def get_scores(self):
```

- **返回**: 包含用户对"喜好度"、"情感价值"和"唤醒度"的评分的字典。
- **说明**: 创建一个GUI窗口供用户使用滑块输入评分。动态更新标签并返回字典形式的评分。

##### 4.format_scale

```python
def format_scale(self, var, value, position, label):
```

- **参数**:
  - `var`: 与滑块关联的tkinter变量。
  - `value`: 滑块的当前值。
  - `position`: 滑块的位置。
  - `label`: 滑块的标签。
- **说明**: 将滑块值四舍五入到最近的整数。

##### 5.save_scores

```python
def save_scores(self, video_path, start_time, end_time, scores):
```

- **参数**:
  - `video_path` (str): 播放视频的路径。
  - `start_time` (datetime): 视频开始播放的时间。
  - `end_time` (datetime): 视频结束的时间。
  - `scores` (dict): `get_scores`返回的评分字典。
- **说明**: 将视频播放信息和用户评分保存到名为'scores.txt'的文本文件中。

#### stim_demo中的调用

选择界面、初始化界面与 `stim_demo.py`中其他范式保持一致

```python
player = mp4play(win=win)
    bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = "COM8"  #  0xdefc                                  # 采集主机端口
    port_addr = None  #  0xdefc
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "mp4play",
        paradigm,
        VSObject=player,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="mp4play",
        lsl_source_id=lsl_source_id,
        online=online,
    )
    while True:
        video_path = player.mp4gain()
        if video_path:
            player.feedback(video_path)
        else:
            break
    ex.run()
```

运行stim_demo.py后将进入视频播放界面：
![image-20240802213231361](https://github.com/hanyilinn/MetaBCI-USTCNeroSpark/blob/master/images/image-20240802213231361.png)

视频播放结束后弹出打分框，我们选用了脑电情绪识别任务常用的3个指标：`liking`表示对短视频的喜好度，`valence`表示该短视频的开心程度，`arousal`表示该短视频的兴奋程度，每个指标满分10分，最低分0分。

![image-20240802213553007](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240802213553007.png)

点击提交后进入下一个视频的播放，若点击关闭则结束实验。

打分的结果将存储于 `scores.txt`文件中，用于后续离线模块训练，存储格式如下：

```python
Play Video Path: D:\MetaBCI-master\tiktok\����2024720-197607.mp4
Starting time: 2024-07-26 17:46:48.469274
Ending time: 2024-07-26 17:47:14.480259
Liking: 5.0
Valence: 5.0
Arousen: 5.0
----------------------------------------
```

其中，`Play Video Path`表示播放的视频路径，`Starting time`表示视频开始播放的时间，`Ending time`表示视频结束播放的时间，`Liking` ,`Valence`, `Arousen`分别记录的是受试者的打分结果。

在播放视频诱导刺激的同时，Muse采集设备也在实时采集数据：

![image-20240802215557908](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240802215557908.png)

### 三、离线训练模块

离线训练模块主要包括两个部分：

1）数据集处理：对于每个视频，从整段脑电数据中截取出该视频的播放开始时间与结束时间之间的脑电数据，进行滤波处理后通过设定的步长和长度截取成若干片段作为数据集。受试者的打分结果为数据集标签。

2）模型的训练：为实现实时处理的低延时性，我们选择基础的多尺度深度卷积模型。

#### 数据集处理

##### 1.butter_bandpass

创建巴特沃斯带通滤波器的系数。

```python
def butter_bandpass(lowcut, highcut, fs, order=5):
```

- **参数**:
  - `lowcut`: 低截止频率。
  - `highcut`: 高截止频率。
  - `fs`: 采样频率。
  - `order`: 滤波器的阶数，默认为5。
- **返回**: 滤波器的系数 `b` 和 `a`。

##### 2. butter_bandpass_filter

对数据应用巴特沃斯带通滤波器。

```python
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
```

- **参数**:
  - `data`: 要滤波的数据。
  - `lowcut`, `highcut`, `fs`, `order`: 与 `butter_bandpass`相同。
- **返回**: 滤波后的数据。

##### 3. ReadCntFile

从CNT文件中读取EEG数据。

```python
def ReadCntFile(cntFileName="EEG.cnt"):
```

- **参数**:
  - `cntFileName`: CNT文件的名称，默认为"EEG.cnt"。
- **返回**: 包含EEG数据的NumPy数组。

##### 4. ReadCntFile_ly

读取CNT文件并根据需要解码特定数量的通道。

```python
def ReadCntFile_ly(cntFileName="EEG.cnt", numProcessSample=2048, isLatestSample=True, file_type='EEG'):
```

- **参数**:
  - `cntFileName`: CNT文件的名称。
  - `numProcessSample`: 要处理的样本数。
  - `isLatestSample`: 是否只读取最新的样本。
  - `file_type`: 数据类型，'EEG' 或 'PPG'。
- **返回**: 包含EEG或PPG数据的NumPy数组。

##### 5. convert_to_timestamp

将时间字符串转换为时间戳。Muse头环默认使用时间戳格式存储数据，故需要将scores.txt中保存的视频开始与结束时间转换为时间戳格式。

```python
def convert_to_timestamp(time_str):
```

- **参数**:
  - `time_str`: 时间字符串。
- **返回**: 转换后的时间戳。

##### 6. parse_txt_file

解析scores.txt文件并提取数据。

```python
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
```

- **参数**:
  - `file_path`: scores.txt文件的路径。
- **返回**: 包含解析数据的列表。

#### 主函数的使用

```python
    step = 128 #数据集划分步长
    window_length = 256 #数据集划分长度
    data_for_exper = []
    datapath = [1722078786118,1722497227125,1722510612662,1722592557872] #4次实验
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
```

#### 模型的初始化与训练

我们使用多尺度卷积网络RACNN-Lite作为模型。划分数据集的前80%为训练集，后20%为测试集。Batch size设置为64，epoch设置为50，学习率设置为0.0001，辍学率设置为0.5。

![image-20240803153330091](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240803153330091.png)

RACNN-Lite包含时间特征提取器、区域特征提取器、非对称特征提取器以及分类器。

时间特征提取器由三个在时间维度上并行的不同尺寸的一维卷积层组成，用于学习多尺度时间特征。卷积核的大小分别为𝑓𝑠/2、𝑓𝑠/4、𝑓𝑠/8，其中𝑓𝑠为采样率。将三个时间特征拼接并同步输入到区域特征提取器和非对称特征提取器。区域特征提取器为一个通道维度的一维卷积层，用于学习区域特征。卷积核大小为2，与输入通道数相同。非对称特征提取器为一个ADL层，用于学习前额叶两通道AF7和AF8的不对称特征。将区域特征和不对称特征拼接后输入到由两个全连接层组成的分类器进行情绪分类。所有卷积层的通道数均为16，两个全连接层的神经元个数分别为32和2。

通过离线模型训练得到个体模型，实现对高/低arousal的二分类。在在线应用阶段，模型输出的0-1之间的概率预测被线性映射为0-100之间的arousal指数，用于在线反馈情绪分数。

### 四、在线测试模块

在线测试模块主要目标是读取脑电采集设备实时写入的数据并输入模型并输出结果。具体实现位于 `Online_Video.py`中。

#### ReadCntFile_ly

根据 `file_type`确定要解码的通道数。使用二进制读取模式打开CNT文件，并读取文件头信息，包括通道数、每样本事件数、采样率等。如果 `isLatestSample`为True，函数将尝试读取文件末尾的最新样本数据。读取的数据被转换为NumPy数组并返回。Muse头环采集的数据实时写入 `/streaming/EEG.net`文件中，因此 `ReadCntFile_ly`在在线测试实验时实时读取该文件中最后若干行的数据以达成实时处理的要求。

```python
def ReadCntFile_ly(cntFileName="EEG.cnt", numProcessSample=2048, isLatestSample=True, file_type='EEG'):
```

- **参数**:
  - `cntFileName`: 要读取的CNT文件名，默认为"EEG.cnt"。
  - `numProcessSample`: 要处理的样本数量，默认为2048。
  - `isLatestSample`: 是否只读取最新的样本，默认为True。
  - `file_type`: 数据类型，可以是'EEG'或'PPG'，默认为'EEG'。
- **返回**: 处理后的数据数组。

#### 主函数

最后的输出即为预测的liking值，转换为百分制形式。

```python
    #加载模型
    model = TSception(num_classes=2, input_size=(1,2,256), sampling_rate=256, num_T=15, num_S=15,hidden=32, dropout_rate=0.5)
    model_path = 'D:\MetaBCI-master\demos\\brainflow_demos\\49model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('已成功加载模型')
    #主循环
    while 1>0:
        try:
            data = ReadCntFile_ly('C:\CBCR\MuseData\Streaming\EEG.cnt', numProcessSample=num)
            if data.shape[0]>=num:
                EEG = data[-256:,1:3].transpose()
                EEG = butter_bandpass_filter(EEG, 0.3, 45, 256, order=6)
                EEG = torch.from_numpy(EEG, ).to(torch.float32).to(device)
                EEG = EEG.unsqueeze(0)
                output = model(EEG)
                probabilities = torch.sigmoid(output)
                value = probabilities[0][1].item().float()
                print("预测的liking值是",value*100,"%")
                time.sleep(0.1)
        except:
            pass
```

如下为一典型的示例结果：
![image-20240803154427150](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20240803154427150.png)
