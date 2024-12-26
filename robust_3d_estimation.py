# -*- coding: utf-8 -*-
"""
robust_3d_estimation.py

功能：
  - 提供 robust_3d_estimation_bbox 函数，根据深度图和边框坐标进行3D点云估计，并去除离群点。

算法细节：
  - 提取子深度图并进行采样；
  - 反投影像素坐标到3D点；
  - 计算中位数并去除离群点。

使用方法：
  - from robust_3d_estimation import robust_3d_estimation_bbox
  - final_xyz, success, inliers = robust_3d_estimation_bbox(depth_img, x1, y1, x2, y2, intr, depth_scale, sample_step=3)
"""

import numpy as np
import pyrealsense2 as rs


def robust_3d_estimation_bbox(depth_img, x1, y1, x2, y2, intr, depth_scale, sample_step=3):
    """
    根据深度图和边框坐标进行3D点云估计，并去除离群点。

    :param depth_img: 深度图 (numpy array)
    :param x1: 边框左上角 x 坐标
    :param y1: 边框左上角 y 坐标
    :param x2: 边框右下角 x 坐标
    :param y2: 边框右下角 y 坐标
    :param intr: 相机内参
    :param depth_scale: 深度图缩放比例，将 z16 转换为米
    :param sample_step: 采样步长，默认值为 3
    :return: 估计的3D坐标 (x, y, z)，是否成功 (bool)，内点集 (numpy array)
    """
    if x1 >= x2 or y1 >= y2:
        return (0, 0, 0), False, None  # 如果边框无效，返回默认值

    sub_depth = depth_img[y1:y2:sample_step, x1:x2:sample_step]  # 提取子深度图
    if sub_depth.size < 10:
        return (0, 0, 0), False, None  # 如果子深度图像素太少，返回默认值

    rows = np.arange(y1, y2, sample_step)  # 生成行索引
    cols = np.arange(x1, x2, sample_step)  # 生成列索引
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, 0), False, None  # 如果行或列索引为空，返回默认值

    grid_c, grid_r = np.meshgrid(cols, rows)  # 生成网格坐标
    depth_flat = sub_depth.reshape(-1).astype(np.float32)  # 展平深度图
    c_flat = grid_c.reshape(-1)  # 展平列坐标
    r_flat = grid_r.reshape(-1)  # 展平行坐标

    valid_mask = (depth_flat > 0)  # 创建有效像素掩码
    if not np.any(valid_mask):
        return (0, 0, 0), False, None  # 如果没有有效像素，返回默认值

    d_val = depth_flat[valid_mask] * depth_scale  # 有效深度值转换为米
    c_val = c_flat[valid_mask]  # 有效列坐标
    r_val = r_flat[valid_mask]  # 有效行坐标
    if len(d_val) < 20:
        return (0, 0, 0), False, None  # 如果有效深度值太少，返回默认值

    points_3d = []
    for i in range(len(d_val)):
        z_m = d_val[i]
        Xp, Yp, Zp = rs.rs2_deproject_pixel_to_point(intr, [int(c_val[i]), int(r_val[i])], z_m)  # 像素坐标反投影到3D点
        points_3d.append([Xp, Yp, Zp])
    points_3d = np.array(points_3d, dtype=np.float32)  # 转换为 numpy array
    if len(points_3d) < 20:
        return (0, 0, 0), False, None  # 如果3D点太少，返回默认值

    # 去除离群点
    med_xyz = np.median(points_3d, axis=0)  # 计算中位数
    dist = np.sqrt(np.sum((points_3d - med_xyz) ** 2, axis=1))  # 计算每个点到中位数的距离
    thr = 0.3  # 距离阈值
    inliers = points_3d[dist < thr]  # 选择内点
    if len(inliers) < 10:
        return (0, 0, 0), False, None  # 如果内点太少，返回默认值

    final_xyz = np.median(inliers, axis=0)  # 计算内点的中位数作为最终估计值
    return tuple(final_xyz), True, inliers  # 返回最终估计值，是否成功，内点集