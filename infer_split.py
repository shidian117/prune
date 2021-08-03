# -*- coding: UTF-8 -*-
"""
模型推理
"""
import os
import config
import cv2
import numpy as np
import time
import paddle.fluid as fluid
import json
from PIL import Image
from PIL import ImageDraw
import shutil
import paddle
paddle.enable_static()

train_parameters = config.init_train_parameters()
yolo_config = train_parameters['yolo_tiny_cfg'] if False else train_parameters['yolo_cfg']
from MobileNet_yolov3_split_client import get_yolo_client
model = get_yolo_client(False, train_parameters['class_dim'], yolo_config['anchors'], yolo_config['anchor_mask'])



label_dict = train_parameters['num_dict']
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]
place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
exe1 = fluid.Executor(place)
exe2 = fluid.Executor(place)
path = train_parameters['freeze_split_dir']  # 'model/freeze_model'
[inference_program1, feed_target_names1, fetch_targets1] = fluid.io.load_inference_model(dirname=path, executor=exe1,
                                                                                      model_filename='__model__',
                                                                                      params_filename='__params__')
[inference_program2, feed_target_names2, fetch_targets2] = fluid.io.load_inference_model(dirname=path, executor=exe2,
                                                                                      model_filename='__model1__',
                                                                                      params_filename='__params1__')


def draw_bbox_image(img, boxes, labels, gt=False):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    color = ['red', 'blue']
    if gt:
        c = color[1]
    else:
        c = color[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, c, width=3)
        draw.text((xmin, ymin), label_dict[int(label)], (255, 255, 0))
    return img


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)

    return img


def read_image(img):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = img
    img = resize_img(origin, yolo_config["input_size"])
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img


def infer(image):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    origin, tensor_img, resized_img = read_image(image)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))
    t1 = time.time()
    batch_outputs_temp = exe1.run(inference_program1,
                            feed={feed_target_names1[0]: tensor_img},
                            fetch_list=fetch_targets1,
                            return_numpy=False)
    #print(batch_outputs_temp[1])
    batch_outputs = exe2.run(inference_program2,
                            feed={feed_target_names2[0]: batch_outputs_temp[0],
                                  feed_target_names2[1]: batch_outputs_temp[1],
                                  feed_target_names2[2]: batch_outputs_temp[2],
                                  feed_target_names2[3]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets2,
                            return_numpy=False)   
    #print(np.array(batch_outputs[0]))               

    

    period = (time.time() - t1) * 1000
    # print("predict cost time:{0}".format("%2.2f ms" % period))
    bboxes = np.array(batch_outputs[0])
    # print(bboxes)
    if bboxes.shape[1] != 6:
        # print("No object found")
        return False, [[], [], [], []], period
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')
    return True, [boxes, labels, scores, bboxes], period


if __name__ == '__main__':
    image_path = 'data/lslm-test/4.jpg'
    img = cv2.imread(image_path)
    flag, [box, label, scores, bboxes], period = infer(img)
    if flag:
        img = draw_bbox_image(img, box, label)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        print('检测到目标')
        cv2.imwrite('result.jpg', img)

    else:
        print(image_path, "没检测出来")
        pass
    print('infer one picture cost {} ms'.format(period))
