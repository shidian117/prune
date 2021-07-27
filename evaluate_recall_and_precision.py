# -*- coding: UTF-8 -*-
"""
完成多分类目标检测任务的召回和精确度计算
"""
from __future__ import division
import numpy as np
import cv2
import json
import config
import os


def mat_inter(box1, box2):
    """
    判断box1与box2代表的两个矩形是否相交
    :param box1:          框1：[x1, y1, x2, y2] 左上角点与右下角点
    :param box2:          框2：格式同框1
    :return: bool型，代表是否相交
    """
    # 判断两个矩形是否相交
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def get_iou(pred_box, gt_box):
    """
    计算给定输入之间的iou，并返回
    :param pred_box:        预测框:[x1, y1, x2, y2] 左上角点与右下角点
    :param gt_box:          实际框:[x1, y1, x2, y2] 左上角点与右下角点
    :return: 1）iou         交并比
    """
    if mat_inter(pred_box, gt_box):
        x01, y01, x02, y02 = pred_box
        x11, y11, x12, y12 = gt_box
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        iou = intersection / (area1 + area2 - intersection)
    else:
        iou = 0
    return iou


def get_iou_idex(pred_box, gt_box_list):
    """
    计算iou, 并返回最大的iou值和对应的gt_box对应的索引值，index
    :param pred_box:          预测框
    :param gt_box_list:       gt_box的列表
    :return:1) max_iou        gt_box_list中与pred_box计算后iou的最大值
            2）max_index      gt_box_list中与pred_box最匹配的gt_box对应的下标
    """
    max_iou = 0  # 开始设置为一个极小值
    max_index = -1  # 开始随机设置一个值

    for i in range(len(gt_box_list)):
        temp_iou = get_iou(pred_box, gt_box_list[i])
        if temp_iou > max_iou:
            max_index = i
            max_iou = temp_iou
    return max_iou, max_index


def get_class_recall_precision(pred_box_list, gt_box_list, score=0.5):
    """
    计算一个类别对应的召回率和精确度
    :param pred_box_list:       该类别预测框的列表
    :param gt_box_list:         该类别实际框的列表
    :param score:               iou的阈值，大于该阈值则认为两个框相交
    :return: 1）recall          召回率
             2）precision       精确度
    """
    true_sample_num = len(gt_box_list)
    # print(true_sample_num)
    pred_class_num = len(pred_box_list)
    gt_box_flag = [0] * true_sample_num
    pred_box_flag = [0] * pred_class_num
    for i in range(pred_class_num):
        iou, index = get_iou_idex(pred_box_list[i], gt_box_list)
        if iou < score:
            continue
        elif gt_box_flag[index] == 0:
            pred_box_flag[i] = 1
            gt_box_flag[index] = 1
    if true_sample_num == 0:
        if pred_class_num == 0:
            recall = -1
            precision = -1
        else:
            recall = -1
            precision = 0
    else:
        if pred_class_num == 0:
            recall = 0
            precision = 0
        else:
            recall = sum(pred_box_flag) / true_sample_num
            precision = sum(pred_box_flag) / pred_class_num
    return recall, precision


def get_sample_recall_precision(pred, gt, class_dic, score=0.5):
    """
    计算每个测试样本中每个类别的召回率、精确度
    :param pred:            预测结果: [[label, score, x1, y1, x2, y2], ...]
    :param gt:              ground_truth：[[gt_label, x1, y1, x2, y2], [gt_label, x1, y1, x2, y2], ...]
    :param class_dic:       类别字典
    :param score:           iou阈值
    :return: 1) sample_recall_precision:     [[label1, recall, precision], [label1, recall, precision], ...]
    """
    sample_recall_precision = []
    box_recall_precision = []
    pred = np.array(pred)
    gt = np.array(gt)
    # 计算安全帽的准召，不考虑颜色
    if len(pred) == 0:
        box_pred = []
        box_score = []
    else:
        box_pred = pred[:, 2:]
        box_score = pred[:, 1]

    if len(gt) == 0:
        box_gt = []
    else:
        box_gt = gt[:, 1:]

    scores = list(zip(box_score, box_pred))
    scores.sort(reverse=True)
    box_pred = [tem_pred for tem_score, tem_pred in scores]
    box_recall, box_presicion = get_class_recall_precision(box_pred, box_gt, score)
    box_recall_precision.append([box_recall, box_presicion])

    for i in class_dic:
        class_pred = []
        class_pred_score = []
        class_gt = []
        for pred_sample in pred:
            if pred_sample[0] == i:
                class_pred.append(pred_sample[2:])
                class_pred_score.append(pred_sample[1])
        # class_pred = [pred_sample[2:] for pred_sample in pred if pred_sample[0] == i]
        # class_pred_score = [pred_sample[1] for pred_sample in pred if pred_sample[0] == i]
        for gt_sample in gt:
            # print(pred_sample)
            if gt_sample[0] == i:
                class_gt.append(gt_sample[1:])
        # class_gt = [gt_sample[1:] for gt_sample in gt if gt_sample[0] == i]
        # if class_gt == []:
        #     continue
        # 对置信度排序，并根据排序结果调整class_pred的顺序
        sorted_score = list(zip(class_pred_score, class_pred))
        sorted_score.sort(reverse=True)
        class_pred = [tem_pred for tem_score, tem_pred in sorted_score]

        # 计算该类别对应的召回和精确度
        class_recall, class_presicion = get_class_recall_precision(class_pred, class_gt, score)
        sample_recall_precision.append([i, class_recall, class_presicion])
    return sample_recall_precision, box_recall_precision
