import math

from psychopy import monitors
import numpy as np
import sys
sys.path.append('D:\MetaBCI-master')
from metabci.brainstim.paradigm import (
    mp4play,
    SSVEP,
    P300,
    MI,
    AVEP,
    SSAVEP,
    paradigm,
    pix2height,
    code_sequence_generate,
)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix
import os
import numpy as np
import random
import cv2
import pygame
import tkinter as tk
from tkinter import ttk
from pygame.locals import *
from psychopy import monitors
from psychopy.tools.monitorunittools import deg2pix
from metabci.brainstim.paradigm  import VisualStim
from metabci.brainstim.framework import Experiment
# 假设我们从这个 IP 地址和端口接收数据
DATA_RECEIVE_IP = '127.0.0.1'
DATA_RECEIVE_PORT = 12345




if __name__ == "__main__":
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix([1920, 1080])  # 显示器的分辨率
    mon.save()
    
    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1920, 1080])
    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    """
    mp4play
    """
    # pygame.display.set_caption('视频播放器')
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
