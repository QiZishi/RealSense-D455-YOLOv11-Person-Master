# -*- coding: utf-8 -*-
"""
posture_classification.py

功能：
  - 提供 refined_action_classification 函数，仅输出 standing、sitting、lying、walking 四种姿态。

算法细节：
  - 如果高宽比大于1.5，且速度大于等于0.2，则判定为 walking
  - 如果高宽比大于1.5，且速度小于0.2，则判定为 standing
  - 如果高宽比大于0.75*0.75，则判定为 sitting
  - 其他情况判定为 lying

使用方法：
  - from posture_classification import refined_action_classification
  - act_str = refined_action_classification(bbox, speed, frame_height=480)
"""


def action_classification(bbox, speed):
    """
    姿态分类函数，根据边框和速度判断目标的姿态
    :param bbox: 边框 (x1, y1, x2, y2)
    :param speed: 目标速度
    :return: 姿态字符串 ('standing', 'sitting', 'lying', 'walking')
    """
    x1, y1, x2, y2 = bbox  # 解包边框坐标
    w = (x2 - x1)  # 计算边框宽度
    h = (y2 - y1)  # 计算边框高度

    ratio = h / float(w + 1e-6)  # 计算高宽比，避免除以零
    if ratio > 1.5:
        if speed >= 0.2:  # 如果速度大于等于 0.2，判定为 walking
            return "walking"
        else:
            return "standing"  # 如果高宽比大于 1.3，且速度小于 0.2，判定为 standing
    elif ratio > 0.75 * 0.75:  # 如果高宽比大于 0.75*0.75，判定为 sitting
        return "sitting"
    else:
        return "lying"  # 其他情况判定为 lying
