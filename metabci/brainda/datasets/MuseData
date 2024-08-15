# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/27
# License: MIT License
"""
Alex Motor imagery dataset.
"""
from typing import Union, Optional, Dict, List, cast
from pathlib import Path
from mne import create_info
from mne.io import RawArray, Raw
from .base import BaseDataset

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

class MuseData(BaseDataset):

    _CHANNELS = [
        'AF7', 'AF8'
    ]
    subject = ["1", "2"]

    def __init__(self, paradigm='mp4'):
        super().__init__(
            dataset_code="MetaBCI_USTCNeroSpark_master",
            subjects=list(range(1, 8)),
            channels=self._CHANNELS,
            events=None,
            srate=256,
            paradigm=paradigm,
        )

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def ReadCntFile(self, cntFileName="EEG.cnt"):
        data = []
        with open(cntFileName, "rb") as f:
            byte = f.read(4)
            num_channel = struct.unpack('i', byte)[0]
            byte = f.read(4)
            iNumEvtPerSample = struct.unpack('i', byte)[0]
            byte = f.read(4)
            iNumSplPerBlock = struct.unpack('i', byte)[0]
            byte = f.read(4)
            sampling_rate = struct.unpack('i', byte)[0]
            byte = f.read(4)
            iDataSize = struct.unpack('i', byte)[0]
            byte = f.read(4)
            resolution = struct.unpack('f', byte)[0]
            byte = f.read(int(num_channel * 10))
            channel_names = struct.unpack('{}s'.format(int(num_channel * 10)), byte)[0]
            channel_names = channel_names.decode('utf-8')
            while byte:
                try:
                    # Do stuff with byte.
                    byte = f.read(num_channel * iDataSize)
                    data.append(struct.unpack(num_channel * 'd', byte))
                except:
                    break

            data = np.array(data)
        return data

    def ReadCntFile_ly(self, cntFileName="EEG.cnt", numProcessSample=2048, isLatestSample=True, file_type='EEG'):
        data = []
        if file_type == 'EEG':
            channel_to_decode = 7
        elif file_type == 'PPG':
            channel_to_decode = 5
        else:
            raise Exception("Wrong data type! Please choose from 'EEG' and 'PPG'.")

        with open(cntFileName, "rb") as f:
            byte = f.read(4)
            num_channel = struct.unpack('i', byte)[0]
            byte = f.read(4)
            iNumEvtPerSample = struct.unpack('i', byte)[0]
            byte = f.read(4)
            iNumSplPerBlock = struct.unpack('i', byte)[0]
            byte = f.read(4)
            sampling_rate = struct.unpack('i', byte)[0]
            byte = f.read(4)
            iDataSize = struct.unpack('i', byte)[0]
            byte = f.read(4)
            resolution = struct.unpack('f', byte)[0]

            if isLatestSample:
                try:
                    f.seek(-num_channel * (numProcessSample + 1) * iDataSize, 2)
                except:
                    data = self.ReadCntFile(cntFileName)
                    # data = np.reshape(data,(-1,7))
                    return data
                while byte:
                    try:
                        # Do stuff with byte.
                        byte = f.read(channel_to_decode * iDataSize)
                        data.append(struct.unpack(channel_to_decode * 'd', byte))
                    except:
                        break
                data = np.array(data)
        return data

    def Process_Online_data(self,num,device):
        data = self.ReadCntFile_ly('C:\CBCR\MuseData\Streaming\EEG.cnt', numProcessSample=num)
        if data.shape[0]>=num:
            EEG = data[-256:,1:3].transpose()
            EEG = self.butter_bandpass_filter(EEG, 0.3, 45, 256, order=6)
            EEG = torch.from_numpy(EEG, ).to(torch.float32).to(device)
            EEG = EEG.unsqueeze(0)
            return EEG

    # 定义一个函数来转换时间字符串为时间戳
    def convert_to_timestamp(self, time_str):
        # 清理字符串，移除干扰字符
        cleaned_str = re.sub(r'^[^0-9]+(.*)', r'\1', time_str)
        print(cleaned_str)
        return round(1000 * (datetime.strptime(cleaned_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()))

    # 定义一个函数来解析scores.txt文本文件
    def parse_txt_file(self, file_path):
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
                    stat_timestamp = self.convert_to_timestamp(stat_time_str)
                    end_timestamp = self.convert_to_timestamp(end_time_str)

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

    def data_path(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        return 0

    def _get_single_subject_data(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ):
        if subject not in self.subjects:
            raise ValueError('Invalid subject {} given'.format(subject))

        filepath = "E:\\metabci_data"
        datapath = [1722078786118, 1722497227125, 1722510612662, 1722592557872]
        step = 128
        window_length = 256
        data_for_exper = []
        subject = cast(int, subject)
        sub_name = str(subject)
        score = 2
        for i in range(len(datapath)):
            data_path = '{:s}/S{:s}/{:s}/EEG.cnt'.format(filepath, sub_name, str(datapath[i]))
            label_path = '{:s}/S{:s}/scores{:s}.txt'.format(filepath, sub_name, str(i+score))

            data = self.ReadCntFile(data_path)
            labels = self.parse_txt_file(label_path)
            data[:, 5] = data[:, 5].astype(np.int64)
            for label in labels:
                if label['valence'] in [0, 1, 2, 3]:
                    flag = 0
                elif label['valence'] in [6, 7, 8, 9]:
                    flag = 1
                else:
                    break
                # 根据 start_time 和 end_time 截取 data 中的时间戳对应的数据
                # print('视频开始时间为：', label['start_time'])
                # print('视频结束时间为：', label['end_time'])
                start_idx = np.searchsorted(data[:, 5], label['start_time'], side='left')
                end_idx = np.searchsorted(data[:, 5], label['end_time'], side='right')
                # 截取 data 中的时间戳为 start_time 和 end_time 之间的数据
                selected_data = data[start_idx:end_idx, 1:3].T
                EEG = self.butter_bandpass_filter(selected_data, 0.3, 45, self.srate, order=6)
                for start in range(0, EEG.shape[1] - window_length + 1, step):
                    window = EEG[:, start:start + window_length]

                    data_for_exper.append((window, flag))  # 假设标签 'a' 对应于所有窗口

        X = np.array([item[0] for item in data_for_exper])  # 数据
        y = np.array([item[1] for item in data_for_exper])  # 标签
        y = y.astype(np.int64)
        print(X.shape)  ##(4667, 2, 256)
        ch_names = [ch_name.upper() for ch_name in self._CHANNELS]
        ch_types = ["eeg"] * 2
        runs = dict()
        info = create_info(ch_names=ch_names,
                           ch_types=ch_types, sfreq=self.srate)
        for i in range(len(data_for_exper)):
            raw = RawArray(
                data=X[i, :, :], info=info
            )##(2, 256)
            runs["run_{:d}".format(i)] = {
                "data": raw,          # 封装 RawArray 对象
                "label": y[i]    # 封装对应的标签
            }##run_4667

        sess = {"session_0": runs}
        subject = {"subject_0": sess}
        return subject

    # def _get_single_subject_data(
    #         self,
    #         subject: Union[str, int],
    #         verbose: Optional[Union[bool, str, int]] = False
    # ):
    #     dests = self.data_path(subject)
    #     montage = make_standard_montage('standard_1005')
    #     montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
    #
    #     sess = dict()
    #     for idx_sess, run_files_path in enumerate(dests):
    #         runs = dict()
    #         raw_temp = []
    #         for idx_run, run_file in enumerate(run_files_path):
    #             raw = read_raw_cnt(run_file,
    #                                eog=['HEO', 'VEO'],
    #                                ecg=['EKG'],
    #                                emg=['EMG'],
    #                                misc=[32, 42, 59, 63],
    #                                preload=True)
    #             raw = upper_ch_names(raw)
    #             raw = raw.pick_types(eeg=True,
    #                                  stim=True,
    #                                  selection=self.channels)
    #             raw.set_montage(montage)
    #             stim_chan = np.zeros((1, raw.__len__()))
    #             # Convert annotation to event
    #             events, _ = \
    #                 mne.events_from_annotations(raw, event_id=(lambda x: int(x)))
    #             # Insert the event to the event channel
    #             for index in range(events.shape[0]):
    #                 if events[index, 2] in self.events_list:
    #                     stim_chan[0, events[index, 0]] = events[index, 2]
    #                     main_event_temp = events[index, 2]
    #                 elif events[index, 2] <= 10 and events[index, 2] % 2 == 1:
    #                     stim_chan[0, events[index, 0]] = self._ALPHA_CODE[
    #                         self.events_key_map[main_event_temp]
    #                     ][int(events[index, 2]/2)]
    #                 else:
    #                     continue
    #             stim_chan_name = ['STI 014']
    #             stim_chan_type = "stim"
    #             stim_info = mne.create_info(
    #                 ch_names=stim_chan_name,
    #                 ch_types=stim_chan_type,
    #                 sfreq=self.srate
    #             )
    #             stim_raw = mne.io.RawArray(
    #                 data=stim_chan,
    #                 info=stim_info
    #             )
    #             # add the stim_chan to data raw object
    #             raw.add_channels([stim_raw])
    #             raw = upper_ch_names(raw)
    #             raw.set_montage(montage)
    #             raw_temp.append(raw)
    #         raw_temp[0].append(raw_temp[1])
    #         raw_temp[2].append(raw_temp[3])
    #         raw_temp[4].append(raw_temp[5])
    #         runs['run_1'] = raw_temp[0]
    #         runs['run_2'] = raw_temp[2]
    #         runs['run_3'] = raw_temp[4]
    #         sess['session_{:d}'.format(idx_sess)] = runs
    #     return sess
