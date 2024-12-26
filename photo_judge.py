# -*- coding: utf-8 -*-
"""
photo_judge.py

功能：
  - 提供 plane_fit_ransac_simplified 函数，基于简化RANSAC判断3D点是否平面化
  - 提供 judge_realperson_or_photo 函数，综合RANSAC结果与深度分布判断是否为 'photo' 或 'real_person'

算法细节：
  - plane_fit_ransac_simplified: 抽样3点拟合平面，内点超过阈值则认为是平面
  - judge_realperson_or_photo: 若RANSAC结果为plane或xyz分布极小，则判定为photo，否则real
  - 若点数太少 (<30) 也判定为photo

使用方法：
  - from photo_judge import judge_realperson_or_photo
  - label = judge_realperson_or_photo(inliers_3d)
"""

import random
import numpy as np


def plane_fit_ransac_simplified(points_3d, dist_thresh=0.02, ratio_thresh=0.8, max_iter=80):
    """
    使用简化的RANSAC算法判断3D点是否平面化
    :param points_3d: 输入的3D点集
    :param dist_thresh: 距离阈值，默认值为0.02
    :param ratio_thresh: 内点比例阈值，默认值为0.8
    :param max_iter: 最大迭代次数，默认值为80
    :return: 如果点集平面化则返回True，否则返回False
    """
    N = len(points_3d)
    if N < 30:
        return True  # 如果点数太少，直接判定为平面
    best_inliers = 0
    target_count = int(ratio_thresh * N)
    idxs = np.arange(N)
    for _ in range(max_iter):
        i1, i2, i3 = random.sample(list(idxs), 3)  # 随机抽样3个点
        p1, p2, p3 = points_3d[i1], points_3d[i2], points_3d[i3]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)  # 计算法向量
        if np.linalg.norm(normal) < 1e-6:
            continue  # 如果法向量太小，跳过
        a, b, c = normal
        d = -np.dot(normal, p1)
        denom = np.linalg.norm(normal)
        dist_all = np.abs(np.dot(points_3d, normal) + d) / denom  # 计算所有点到平面的距离
        inliers = np.sum(dist_all <= dist_thresh)  # 计算内点数量
        if inliers > best_inliers:
            best_inliers = inliers
            if best_inliers >= target_count:
                return True  # 如果内点数量超过阈值，判定为平面
    ratio_val = best_inliers / float(N)
    return (ratio_val >= ratio_thresh)  # 根据内点比例判定是否为平面


def judge_realperson_or_photo(inliers_3d):
    """
    判断是否为真人或照片
    :param inliers_3d: 输入的3D点集
    :return: 'photo' 或 'person'
    """

    # RANSAC
    plane_flag = plane_fit_ransac_simplified(inliers_3d, dist_thresh=0.02, ratio_thresh=0.8, max_iter=80)
    if plane_flag:
        return "photo"  # 如果RANSAC结果为平面，判定为照片

    # 分布检查
    xyz_min = np.min(inliers_3d, axis=0)
    xyz_max = np.max(inliers_3d, axis=0)
    diff_xyz = xyz_max - xyz_min
    if diff_xyz[0] < 0.02 and diff_xyz[1] < 0.02 and diff_xyz[2] < 0.02:
        return "photo"  # 如果xyz分布极小，判定为照片

    return "person"  # 否则判定为真人
