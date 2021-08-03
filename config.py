# -*- coding: UTF-8 -*-
"""
配置参数
"""
import os
import logging
import codecs

train_parameters = {
    "data_dir": "data",
    "train_list": "train.txt",
    "eval_list": "eval.txt",
    "use_filter": False,
    "class_dim": -1,
    "label_dict": {},
    "num_dict": {},
    "image_count": -1,
    "print_params": True,
    "continue_train": False,  # 是否加载前一次的训练参数，接着训练
    "pretrained": True,
    "pretrained_model_dir": "model/pretrained_model",
    "save_model_dir": "./model/yolo_model",
    "freeze_split_dir": "split_freeze_model/",
    "model_prefix": "yolo-v3",
    "freeze_dir": "freeze_model/",
    "use_tiny": False,  # 是否使用 裁剪 tiny 模型
    "max_box_num": 50,  # 一幅图上最多有多少个目标
    "num_epochs": 130,
    "train_batch_size": 8,  # 对于完整 yolov3，每一批的训练样本不能太多，内存会炸掉
    "use_gpu": True,
    #####mobilenetv1-yolov3裁剪配置参数
    #conv2d_10.w_0,conv2d_11.w_0,conv2d_12.w_0,conv2d_13.w_0,conv2d_14.w_0,    0.2,0.2,0.2,0.2,0.2,
    "pruned_params": "conv2d_24.w_0,conv2d_25.w_0,conv2d_26.w_0,yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights",
    "pruned_ratios": "0.2,0.3,0.4,0.5,0.5,0.5,0.5,0.5,0.5,0.7,0.7,0.7,0.7,0.7,0.7,0.8,0.8,0.8,0.8,0.8,0.8",
    "split_model_size":[[21, 14, 14],[21, 28, 28],[21, 56, 56]],
    ###############################
    "yolo_cfg": {
        "input_size": [3, 448, 448],  # 原版的边长大小为608，为了提高训练速度和预测速度，此处压缩为448
        "anchors": [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
        "anchor_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    },
    "yolo_tiny_cfg": {
        "input_size": [3, 256, 256],
        "anchors": [6, 8, 13, 15, 22, 34, 48, 50, 81, 100, 205, 191],
        "anchor_mask": [[3, 4, 5], [0, 1, 2]]
    },
    "ignore_thresh": 0.7,
    "mean_rgb": [127.5, 127.5, 127.5],
    "mode": "train",
    "multi_data_reader_count": 4,
    "apply_distort": True,
    "nms_top_k": 300,
    "nms_pos_k": 300,
    "valid_thresh": 0.1,
    "nms_thresh": 0.45,
    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.5
    },
    "opt_strategy": {
        "learning_rate": 0.001,
        "lr_epochs": [40, 80, 110],
        
    },
    "early_stop": {
        "sample_frequency": 50,
        "rise_limit": 10,
        "min_loss": 0.0005,
        "min_curr_map": 0.84
    }
}


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """
    file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_list'])
    label_list = os.path.join(train_parameters['data_dir'], "label_list")
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            train_parameters['num_dict'][index] = line.strip()
            train_parameters['label_dict'][line.strip()] = index
            index += 1
        train_parameters['class_dim'] = index
        # print(train_parameters['class_dim'])
    with codecs.open(file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)
        # print(train_parameters['image_count'])
    return train_parameters

def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger
