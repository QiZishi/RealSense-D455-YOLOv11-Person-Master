# -*- coding: utf-8 -*-
"""
motion_tracker.py

功能：
  - 实现 Track 和 MultiObjectTracker 类，用于管理多目标跟踪
  - 使用卡尔曼滤波对真人目标进行 predict/update，若目标为照片则不再更新

算法细节：
  - Track 包含 KalmanFilter3D 对象 (或 Photo 标志)
  - MultiObjectTracker 维护多个 Track，并根据距离或 IOU 进行匹配

使用方法：
  - from motion_tracker import MultiObjectTracker
  - mot = MultiObjectTracker(max_missing_time=2.0)
  - mot.predict_all(current_time)
  - mot.update_tracks(detections, current_time)
"""

import math
import numpy as np
from utils.kalman_3d import KalmanFilter3D
from scipy.optimize import linear_sum_assignment


def box_iou(a, b):
    """
    计算两个边框的 IOU（交并比）
    :param a: 边框 a (x1, y1, x2, y2)
    :param b: 边框 b (x1, y1, x2, y2)
    :return: IOU 值
    """
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    if areaA <= 0 or areaB <= 0:
        return 0.0
    return interArea / float(areaA + areaB - interArea)


class Track:
    def __init__(self, track_id, init_pos3d, bbox, init_time, is_photo=False):
        """
        初始化 Track 对象
        :param track_id: 跟踪 ID
        :param init_pos3d: 初始 3D 位置 (x, y, z)
        :param bbox: 初始边框 (x1, y1, x2, y2)
        :param init_time: 初始时间
        :param is_photo: 是否为照片，默认值为 False
        """
        self.track_id = track_id  # 跟踪 ID
        self.kf = KalmanFilter3D(init_pos3d, init_time)  # 初始化卡尔曼滤波器
        self.bbox = bbox  # 边框
        self.last_update_time = init_time  # 上次更新时间
        self.is_photo = is_photo  # 是否为照片

    def predict(self, current_time):
        """
        预测下一时刻的状态
        :param current_time: 当前时间
        """
        if not self.is_photo:  # 如果不是照片
            self.kf.predict(current_time)  # 进行卡尔曼预测

    def update(self, pos3d, bbox, current_time, is_photo):
        """
        更新状态
        :param pos3d: 3D 位置 (x, y, z)
        :param bbox: 边框 (x1, y1, x2, y2)
        :param current_time: 当前时间
        :param is_photo: 是否为照片
        """
        if not self.is_photo:  # 如果不是照片
            self.kf.update(np.array(pos3d, dtype=np.float32))  # 更新卡尔曼滤波器
            self.bbox = bbox  # 更新边框
            self.last_update_time = current_time  # 更新最后更新时间
        if is_photo:  # 如果是照片
            self.is_photo = True  # 标记为照片

    def get_state(self):
        """
        获取当前状态
        :return: 位置 (x, y, z) 和速度 (vx, vy, vz)，边框，是否为照片
        """
        if self.is_photo:  # 如果是照片
            return (0, 0, 0), (0, 0, 0), self.bbox, True  # 返回照片状态
        else:
            pos, vel = self.kf.get_state()  # 获取卡尔曼滤波器状态
            return pos, vel, self.bbox, False  # 返回状态


class MultiObjectTracker:
    def __init__(self, max_missing_time=5.0):
        """
        初始化多目标跟踪器
        :param max_missing_time: 最大丢失时间，默认值为 2.0 秒
        """
        self.tracks = {}  # 跟踪对象字典
        self.next_id = 0  # 下一个跟踪 ID
        self.max_missing_time = max_missing_time  # 最大丢失时间

    def create_track(self, pos3d, bbox, current_time, is_photo):
        """
        创建新的跟踪对象
        :param pos3d: 3D 位置 (x, y, z)
        :param bbox: 边框 (x1, y1, x2, y2)
        :param current_time: 当前时间
        :param is_photo: 是否为照片
        """
        t = Track(self.next_id, pos3d, bbox, current_time, is_photo)  # 创建 Track 对象
        self.tracks[self.next_id] = t  # 添加到跟踪字典
        self.next_id += 1  # 更新下一个跟踪 ID

    def predict_all(self, current_time):
        """
        预测所有跟踪对象的下一时刻状态
        :param current_time: 当前时间
        """
        for tid in self.tracks:  # 遍历所有跟踪对象
            self.tracks[tid].predict(current_time)  # 预测状态

    def remove_lost_tracks(self, current_time):
        """
        移除丢失的跟踪对象
        :param current_time: 当前时间
        """
        remove_ids = []  # 要移除的跟踪 ID 列表
        for tid, t in self.tracks.items():  # 遍历所有跟踪对象
            if (current_time - t.last_update_time) > self.max_missing_time:  # 如果丢失时间超过最大丢失时间
                remove_ids.append(tid)  # 添加到移除列表
        for rid in remove_ids:  # 遍历移除列表
            del self.tracks[rid]  # 从跟踪字典中删除

    def update_tracks(self, detections, current_time):
        """
        更新跟踪对象
        :param detections: 检测结果 [(pos3d, bbox, is_photo), ...]
        :param current_time: 当前时间
        """
        if len(detections) == 0 and len(self.tracks) == 0:  # 如果没有检测结果且没有跟踪对象
            return
        track_ids = list(self.tracks.keys())  # 获取所有跟踪 ID
        if len(track_ids) == 0 and len(detections) > 0:  # 如果没有跟踪对象但有检测结果
            for det in detections:  # 遍历检测结果
                pos3d, bx, is_pho = det
                self.create_track(pos3d, bx, current_time, is_pho)  # 创建新的跟踪对象
            return
        elif len(detections) == 0:  # 如果没有检测结果
            return

        N = len(track_ids)  # 跟踪对象数量
        M = len(detections)  # 检测结果数量
        cost_matrix = np.zeros((N, M), dtype=np.float32)  # 成本矩阵
        for i, tid in enumerate(track_ids):  # 遍历所有跟踪对象
            pos_est, _, box_est, is_pho_t = self.tracks[tid].get_state()  # 获取跟踪对象状态
            for j, det in enumerate(detections):  # 遍历所有检测结果
                pos_det, bx_det, is_pho_d = det
                if is_pho_t and is_pho_d:  # 如果都是照片
                    iouVal = box_iou(box_est, bx_det)  # 计算 IOU
                    cost_matrix[i, j] = 1.0 - iouVal  # 成本为 1 - IOU
                elif is_pho_t and not is_pho_d:  # 如果跟踪对象是照片但检测结果不是
                    cost_matrix[i, j] = 9999  # 设置高成本
                elif not is_pho_t and is_pho_d:  # 如果跟踪对象不是照片但检测结果是
                    cost_matrix[i, j] = 9999  # 设置高成本
                else:  # 如果都是真人
                    dx = pos_est[0] - pos_det[0]  # 计算 x 方向距离
                    dy = pos_est[1] - pos_det[1]  # 计算 y 方向距离
                    dz = pos_est[2] - pos_det[2]  # 计算 z 方向距离
                    cost_matrix[i, j] = math.sqrt(dx * dx + dy * dy + dz * dz)  # 成本为 3D 距离

        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # 线性分配
        used_tracks = set()  # 已使用的跟踪对象
        used_dets = set()  # 已使用的检测结果

        for r, c in zip(row_ind, col_ind):  # 遍历分配结果
            pos3d_d, bx_d, is_pho_d = detections[c]
            _, _, box_est, is_pho_t = self.tracks[track_ids[r]].get_state()
            if is_pho_t:  # 如果是照片
                if cost_matrix[r, c] < 0.7:  # 如果成本小于 0.7
                    used_tracks.add(track_ids[r])  # 添加到已使用的跟踪对象
                    used_dets.add(c)  # 添加到已使用的检测结果
            else:  # 如果是真人
                if cost_matrix[r, c] < 1.5:  # 如果成本小于 1.5
                    used_tracks.add(track_ids[r])  # 添加到已使用的跟踪对象
                    used_dets.add(c)  # 添加到已使用的检测结果

        for r, c in zip(row_ind, col_ind):  # 遍历分配结果
            if (track_ids[r] in used_tracks) and (c in used_dets):  # 如果跟踪对象和检测结果都已使用
                pos3d_d, bx_d, is_pho_d = detections[c]
                self.tracks[track_ids[r]].update(pos3d_d, bx_d, current_time, is_pho_d)  # 更新跟踪对象

        unmatched_dets = [idx for idx in range(M) if idx not in used_dets]  # 未匹配的检测结果
        for ud in unmatched_dets:  # 遍历未匹配的检测结果
            pos3d_d, box_d, is_pho_d = detections[ud]
            self.create_track(pos3d_d, box_d, current_time, is_pho_d)  # 创建新的跟踪对象
