# -*- coding: utf-8 -*-
"""
remove_bg.py

功能：
  - 提供 remove_background 函数，用于根据深度阈值（bg_thresh）将背景像素置为0。

算法细节：
  - 将深度图（单位：z16）转换为米（depth_img * depth_scale）；
  - 若深度大于 bg_thresh 则视为背景置0，否则保留原深度值。

使用方法：
  - from remove_bg import remove_background
  - depth_bg = remove_background(depth_img, depth_scale, bg_thresh=3.0)
"""

import numpy as np


def remove_background(depth_img, depth_scale, bg_thresh=3.0):
    """
    将大于 bg_thresh 的像素直接置0.

    :param depth_img: z16格式的深度图 (numpy array)
    :param depth_scale: 将z16转换为米的scale
    :param bg_thresh: 阈值(米), 大于此认为是背景，默认值为3.0米
    :return: 处理后的深度图 (numpy array)
    """
    depth_m = depth_img * depth_scale  # 将深度图转换为米
    valid_mask = (depth_m > 0) & (depth_m < bg_thresh)  # 创建有效像素掩码，深度在0到bg_thresh之间
    out = np.zeros_like(depth_img, dtype=np.uint16)  # 初始化输出图像，所有像素值为0
    out[valid_mask] = depth_img[valid_mask]  # 保留有效像素的原始深度值
    return out  # 返回处理后的深度图