# -*- coding: utf-8 -*-
"""
kalman_3d.py

功能：
  - 提供 KalmanFilter3D 类，对目标在三维空间 (x, y, z) 进行卡尔曼滤波跟踪，状态含 vx, vy, vz。

算法细节：
  - 采用简单的匀速模型 F，观测矩阵 H = [I(3x3), 0(3x3)]；
  - 支持 predict(current_time) 与 update(z)。

使用方法：
  - from kalman_3d import KalmanFilter3D
  - kf = KalmanFilter3D(init_pos, init_time)
  - kf.predict(t), kf.update(z)
"""

import numpy as np

class KalmanFilter3D:
    def __init__(self, init_pos, init_time, process_noise=0.2, measurement_noise=0.1):
        """
        初始化卡尔曼滤波器
        :param init_pos: 初始位置 (x, y, z)
        :param init_time: 初始时间
        :param process_noise: 过程噪声，默认值为0.2
        :param measurement_noise: 测量噪声，默认值为0.1
        """
        # 初始状态 [x, y, z, vx, vy, vz]
        self.state = np.array([init_pos[0], init_pos[1], init_pos[2], 0, 0, 0], dtype=np.float32)
        # 初始协方差矩阵
        self.P = np.eye(6, dtype=np.float32) * 0.1

        # 状态转移矩阵 F
        self.F = np.eye(6, dtype=np.float32)
        # 观测矩阵 H
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1

        # 过程噪声协方差矩阵 Q
        self.Q = np.eye(6, dtype=np.float32) * process_noise
        # 测量噪声协方差矩阵 R
        self.R = np.eye(3, dtype=np.float32) * measurement_noise

        # 上次更新时间
        self.last_time = init_time

    def predict(self, current_time):
        """
        预测下一时刻的状态
        :param current_time: 当前时间
        """
        # 计算时间差
        dt = current_time - self.last_time
        self.last_time = current_time
        # 更新状态转移矩阵 F
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        # 预测下一时刻的状态
        self.state = self.F @ self.state
        # 更新协方差矩阵 P
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        更新状态
        :param z: 测量值 (x, y, z)
        """
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # 更新状态
        y = z - (self.H @ self.state)
        self.state = self.state + K @ y
        # 更新协方差矩阵 P
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        """
        获取当前状态
        :return: 位置 (x, y, z) 和速度 (vx, vy, vz)
        """
        return self.state[:3], self.state[3:6]