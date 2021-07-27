# -*- coding:utf-8 -*-
"""
模型训练
"""
from __future__ import division
import os
import time
import json
import config
import cv2
import numpy as np
from MobileNet_yolov3 import get_yolo
import paddle.fluid as fluid
from reader import single_custom_reader
from PIL import Image
from evaluate_recall_and_precision import get_sample_recall_precision
from paddleslim.prune import Pruner
from paddleslim.analysis import flops
from learning_rate import exponential_with_warmup_decay

logger = config.init_log_config()
train_parameters = config.init_train_parameters()
# print(train_parameters)
yolo_config = train_parameters['yolo_tiny_cfg'] if train_parameters["use_tiny"] else train_parameters["yolo_cfg"]
# yolo_config = train_parameters["yolo_cfg"]
# 先把验证集整个读进来，减少每次重新读取带来不必要的时间浪费
data_dir = train_parameters["data_dir"]
place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
exe = fluid.Executor(place)
label_dict = train_parameters['num_dict']
label_dict = dict(zip(label_dict.values(), label_dict.keys()))
test_file_path = os.path.join(data_dir, train_parameters['eval_list'])
val_data = []
print(label_dict)
with open(test_file_path, 'r') as f:
    lines = f.readlines()
    for sample in range(len(lines)):
        line = lines[sample].split('\t')
        pic_path = os.path.join(data_dir, line[0])
        gt_list = []
        for gt_info in line[1:]:
            if len(gt_info) <= 1:
                continue

            object_info = json.loads(gt_info)
            bbox = object_info['coordinate']
            # print(bbox)
            gt_list.append([label_dict[object_info['value']], bbox[0][0],
                            bbox[0][1], bbox[1][0], bbox[1][1]])
        if len(gt_list) == 0:
            continue
        img = cv2.imread(pic_path)
        try:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except:
            print(pic_path)
            continue
        input_w, input_h = img.size[0], img.size[1]
        image_shape = np.array([input_h, input_w], dtype='int32')
        img = img.resize(yolo_config["input_size"][1:], Image.BILINEAR)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
        img -= 127.5
        img *= 0.007843
        img = img[np.newaxis, :]
        val_data.append([img, image_shape, gt_list])
    print(len(val_data))


def eval_recall(program, fetch_list):
    """
    eval
    :param program:
    :param fetch_list:
    :return:
    """
    recall_category = [[0, 0] for i in range(len(label_dict))]
    precision_category = [[0, 0] for i in range(len(label_dict))]
    recall = []
    precision = []
    recall_box = []
    precision_box = []
    
    time_cost = []
    print(len(val_data))
    for i in val_data:
        t1 = time.time()
        temp_image = i[0]
        temp_image_shape = i[1]
        temp_gt = i[2]
        box = exe.run(program, feed={'img': temp_image,
                                     'image_shape': temp_image_shape[np.newaxis, :]}, fetch_list=fetch_list,
                      return_numpy=False)
        bboxes = np.array(box[0])
        if bboxes.shape[1] != 6:
            labels, scores, boxes = [], [], []
        else:
            labels = bboxes[:, 0].astype('int32')
            scores = bboxes[:, 1].astype('float32')
            boxes = bboxes[:, 2:].astype('float32')
        pred_time = time.time() - t1
        time_cost.append(pred_time)
        pred_list = []
        if len(labels):
            for num in range(len(boxes)):
                pred_list.append([labels[num], scores[num], boxes[num][0],
                                  boxes[num][1], boxes[num][2], boxes[num][3]])
        sample_result, box_results = get_sample_recall_precision(pred_list, temp_gt, label_dict.values(), 0.5)

        for sample_class_result in sample_result:
            if sample_class_result[1] != -1:
                recall.append(sample_class_result[1])
                recall_category[sample_class_result[0]][0] += sample_class_result[1]
                recall_category[sample_class_result[0]][1] += 1
            if sample_class_result[2] != -1:
                precision.append(sample_class_result[2])
                precision_category[sample_class_result[0]][0] += sample_class_result[2]
                precision_category[sample_class_result[0]][1] += 1

    mean_recall = sum(recall) / (len(recall) + 0.0001)
    mean_precision = sum(precision) / (len(precision) + 0.0001)
    if mean_recall + mean_precision == 0:
        f1_score = 0
    else:
        f1_score = 2 * (mean_recall * mean_precision) / (mean_recall + mean_precision)

    mean_time = sum(time_cost[3:]) / len(time_cost[3:])
    recall_category = [item[0] / (item[1] + 0.0001) for item in recall_category]  # 各个类别的recall
    precision_category = [item[0] / (item[1] + 0.0001) for item in precision_category]  # 各个类别的precision

    result = {}
    result["f1_score"] = f1_score
    result["mean_recall"] = mean_recall
    result["mean_precision"] = mean_precision
    result["recall_category"] = recall_category
    result["precision_category"] = precision_category
    result["mean_time"] = mean_time

    return result


def create_tmp_var(programe, name, dtype, shape):
    """
    create_tmp_var
    :param programe:
    :param name:
    :param dtype:
    :param shape:
    :return:
    """
    return programe.current_block().create_var(name=name, dtype=dtype, shape=shape)


def split_by_anchors(gt_box, gt_label, image_size, down_ratio, yolo_anchors):
    """
    将 ground truth 的外接矩形框分割成一个一个小块，类似 seg-link 中的做法
    :param gt_box: 真实外接矩形框，按照 [x, y, w, h] 排布的二维 list，第一维是batch，实际的值都是除以了原始图片尺寸的比例值
    :param gt_label: 真实的类别标签二维 Lise，第一维是batch
    :param image_size: 训练图片的尺寸，[h, w]
    :param down_ratio: int 类型，下采样比例，也暗示现在的特征图被分成多大
    :param yolo_anchors: 当前批次的anchors
    :return:
    """

    gt_box = np.array(gt_box)
    gt_label = np.array(gt_label)
    image_size = np.array(image_size)
    down_ratio = np.array(down_ratio)[0]
    yolo_anchors = np.array(yolo_anchors)
    # print('gt_box shape:{0} gt_label:{1} image_size:{2} down_ratio:{3} yolo_anchors:{4}'
    #       .format(gt_box.shape, gt_label.shape, image_size, down_ratio, yolo_anchors))
    tolerant_ratio = 1.85
    ret_shift_box = np.zeros(gt_box.shape, gt_box.dtype)
    ret_shift_label = np.zeros(gt_label.shape, gt_label.dtype)
    max_bbox = 0

    for n in range(gt_box.shape[0]):
        current_index = 0
        for i in range(gt_box.shape[1]):
            bbox_h = gt_box[n, i, 3] * image_size[0]
            if bbox_h <= 0.1:
                break
            for anchor_h in yolo_anchors[::2]:
                h_d_s = bbox_h / anchor_h
                s_d_h = anchor_h / bbox_h
                if h_d_s <= tolerant_ratio and s_d_h <= tolerant_ratio:
                    ret_shift_box[n, current_index] = gt_box[n, i]
                    ret_shift_label[n, current_index] = gt_label[n, i]
                    current_index += 1
                    if i > max_bbox:
                        max_bbox = i
                    break

    return [ret_shift_box, ret_shift_label]


def optimizer_momentum_setting():
    """
    momentum
    :return:
    """
    # lr_steps = train_parameters['opt_strategy']['lr_epochs']*(train_parameters["image_count"]//train_parameters["train_batch_size"])
    lr_steps = [item * (train_parameters["image_count"]//train_parameters["train_batch_size"]) for item in train_parameters['opt_strategy']['lr_epochs']]
    # lr_steps = [70000, 120000]
    print(lr_steps)
    learning_rate = train_parameters['opt_strategy']['learning_rate']
    boundaries = lr_steps
    gamma = 0.1
    step_num = len(lr_steps)
    values = [learning_rate * (gamma ** i) for i in range(step_num + 1)]

    lr = exponential_with_warmup_decay(
        learning_rate=learning_rate,
        boundaries=boundaries,
        values=values,
        warmup_iter=1000,
        warmup_factor=0.)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(0.0005),
        momentum=0.9)

    return optimizer, lr


def build_program_with_feeder(main_prog, startup_prog, place=None, istrain=True):
    """
    build_program_with_feeder
    :param main_prog:
    :param startup_prog:
    :param place:
    :param istrain:
    :return:
    """
    max_box_num = train_parameters['max_box_num']
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')
        with fluid.unique_name.guard():
            model = get_yolo(ues_tiny, train_parameters['class_dim'], yolo_config['anchors'],
                             yolo_config['anchor_mask'])

            outputs = model.net(img)
            if istrain:
                gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 4], dtype='float32')
                gt_label = fluid.layers.data(name='gt_label', shape=[max_box_num], dtype='int32')
                feeder = fluid.DataFeeder(feed_list=[img, gt_box, gt_label], place=place, program=main_prog)
                reader = single_custom_reader(os.path.join(data_dir, train_parameters['train_list']),
                                              train_parameters['data_dir'],
                                              yolo_config['input_size'], 'train')
                loss = get_loss(model, outputs, gt_box, gt_label, main_prog)
                return feeder, reader, loss
            else:
                boxes = []
                scores = []
                image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='int32')
                downsample_ratio = model.get_downsample_ratio()
                for i, out in enumerate(outputs):
                    box, score = fluid.layers.yolo_box(
                        x=out,
                        img_size=image_shape,
                        anchors=model.get_yolo_anchors()[i],
                        class_num=model.get_class_num(),
                        conf_thresh=train_parameters['valid_thresh'],
                        downsample_ratio=downsample_ratio,
                        name="yolo_box_" + str(i))
                    boxes.append(box)
                    scores.append(fluid.layers.transpose(score, perm=[0, 2, 1]))
                    downsample_ratio //= 2

                pred = fluid.layers.multiclass_nms(
                    bboxes=fluid.layers.concat(boxes, axis=1),
                    scores=fluid.layers.concat(scores, axis=2),
                    score_threshold=0.1,
                    nms_top_k=train_parameters['nms_top_k'],
                    keep_top_k=train_parameters['nms_pos_k'],
                    nms_threshold=train_parameters['nms_thresh'],
                    background_label=-1,
                    name="multiclass_nms")
                return pred


def get_loss(model, outputs, gt_box, gt_label, main_prog):
    """
    get_loss
    :param model:
    :param outputs:
    :param gt_box:
    :param gt_label:
    :param main_prog:
    :return:
    """
    losses = []
    downsample_ratio = model.get_downsample_ratio()
    with fluid.unique_name.guard('train'):
        for i, out in enumerate(outputs):
            if train_parameters['use_filter']:
                ues_tiny = train_parameters['use_tiny']
                yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
                train_image_size_tensor = fluid.layers.assign(np.array(yolo_config['input_size'][1:]).astype(np.int32))
                down_ratio = fluid.layers.fill_constant(shape=[1], value=downsample_ratio, dtype=np.int32)
                yolo_anchors = fluid.layers.assign(np.array(model.get_yolo_anchors()[i]).astype(np.int32))
                filter_bbox = create_tmp_var(main_prog, None, gt_box.dtype, gt_box.shape)
                filter_label = create_tmp_var(main_prog, None, gt_label.dtype, gt_label.shape)
                fluid.layers.py_func(func=split_by_anchors,
                                     x=[gt_box, gt_label, train_image_size_tensor, down_ratio, yolo_anchors],
                                     out=[filter_bbox, filter_label])
            else:
                filter_bbox = gt_box
                filter_label = gt_label

            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=filter_bbox,
                gt_label=filter_label,
                anchors=model.get_anchors(),
                anchor_mask=model.get_anchor_mask()[i],
                class_num=model.get_class_num(),
                ignore_thresh=train_parameters['ignore_thresh'],
                use_label_smooth=False,  # 对于类别不多的情况，设置为 False 会更合适一些，不然 score 会很小
                downsample_ratio=downsample_ratio)
            losses.append(fluid.layers.reduce_mean(loss))
            downsample_ratio //= 2
        loss = sum(losses)
        optimizer, lr = optimizer_momentum_setting()
        optimizer.minimize(loss)
        return [loss, lr]


def load_pretrained_params(exe, program):
    """
    load_pretrained_params
    :param exe:
    :param program:
    :return:
    """
    if train_parameters['continue_train'] and os.path.exists(train_parameters['save_model_dir']):
        logger.info('load param from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=train_parameters['save_model_dir'],
                                   main_program=program)
    elif train_parameters['pretrained'] and os.path.exists(train_parameters['pretrained_model_dir']):
        logger.info('load param from pretrained model')

        def if_exist(var):
            """
            if_exist
            :param var:
            :return:
            """
            return os.path.exists(os.path.join(train_parameters['pretrained_model_dir'], var.name))

        fluid.io.load_vars(exe, train_parameters['pretrained_model_dir'], main_program=program,
                           predicate=if_exist)


def train():
    """
    train
    :return:
    """
    logger.info("start train YOLOv3, train params:%s", str(train_parameters))

    logger.info("create place, use gpu:" + str(train_parameters['use_gpu']))

    logger.info("build network and program")

    scope = fluid.global_scope()
    train_program = fluid.Program()
    start_program = fluid.Program()
    test_program = fluid.Program()

    feeder, reader, loss = build_program_with_feeder(train_program, start_program, place)

    pred = build_program_with_feeder(test_program, start_program, istrain=False)

    test_program = test_program.clone(for_test=True)

    train_fetch_list = [loss[0].name, loss[1].name]

    exe.run(start_program, scope=scope)

    load_pretrained_params(exe, train_program)

    ########################################
    # 此处将模型参数打印出来，从中挑选需要裁剪的参数，并写在config.py相应的配置处
    if train_parameters['print_params']:
        param_delimit_str = '-' * 20 + "All parameters in current graph" + '-' * 20
        print(param_delimit_str)
        for block in train_program.blocks:
            for param in block.all_parameters():
                print("parameter name: {}\tshape: {}".format(param.name,
                                                             param.shape))
        print('-' * len(param_delimit_str))

    pruned_params = train_parameters['pruned_params'].strip().split(",") #此处也可以通过写正则表达式匹配参数名
    logger.info("pruned params: {}".format(pruned_params))
    pruned_ratios = [float(n) for n in train_parameters['pruned_ratios'].strip().split(",")]
    logger.info("pruned ratios: {}".format(pruned_ratios))

    logger.info("build executor and init params")

    pruner = Pruner()
    train_program = pruner.prune(
        train_program,
        scope,
        params=pruned_params,
        ratios=pruned_ratios,
        place=place,
        only_graph=False)[0]

    base_flops = flops(test_program)
    test_program = pruner.prune(
        test_program,
        scope,
        params=pruned_params,
        ratios=pruned_ratios,
        place=place,
        only_graph=True)[0]
    pruned_flops = flops(test_program)

    print("pruned FLOPS: {}".format(
        float(base_flops - pruned_flops) / base_flops))

    stop_strategy = train_parameters['early_stop']
    rise_limit = stop_strategy['rise_limit']
    # sample_freq = stop_strategy['sample_frequency']
    # min_curr_map = stop_strategy['min_curr_map']
    min_loss = stop_strategy['min_loss']
    # stop_train = False
    rise_count = 0
    total_batch_count = 0
    current_best_f1 = 0.0
    train_temp_loss = 0
    current_best_pass = 0
    current_best_box_pass = 0
    current_best_recall = 0
    current_best_precision = 0
    current_best_box_recall = 0
    current_best_box_precision = 0
    current_best_box_f1 = 0
    max_iters = train_parameters["num_epochs"]*(train_parameters["image_count"]//train_parameters["train_batch_size"])
    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: {}, start read image".format(pass_id))
        batch_id = 0
        total_loss = 0.0
        for batch_id, data in enumerate(reader()):
            t1 = time.time()
            loss_ = exe.run(train_program, feed=feeder.feed(data), fetch_list=train_fetch_list)
            period = time.time() - t1
            loss = np.mean(np.array(loss_[0]))
            total_loss += loss
            batch_id += 1
            total_batch_count += 1
            iters = pass_id * (train_parameters['image_count'] // train_parameters['train_batch_size']) + batch_id

            if iters % 500 == 0:
                logger.info("pass {}, iters {}, loss {}, lr {}, time {}".format(pass_id, iters, loss, loss_[1],
                                                                                "%2.2f sec" % period))
        pass_mean_loss = total_loss / batch_id
        logger.info("pass {0} train result, current pass mean loss: {1}".format(pass_id, pass_mean_loss))
        # 采用每训练完一轮停止办法，可以调整为更精细的保存策略
        # print(test_program.global_block().vars)
        if pass_id > 15:
            result = eval_recall(test_program, [pred.name])
            pass_f1 = result["f1_score"]
            pass_recall = result["mean_recall"]
            pass_precision = result["mean_precision"]
            recall_category = result["recall_category"]
            precision_category = result["precision_category"]
            mean_time = result["mean_time"]

            logger.info("{} epoch current pass f1 is {}".format(pass_id, pass_f1))
            logger.info("{} epoch current pass recall is {}".format(pass_id, pass_recall))
            logger.info("{} epoch current pass precision is {}".format(pass_id, pass_precision))
            logger.info("infer one picture time cost {} ms".format(mean_time*1000))

            if pass_f1 >= current_best_f1:
                logger.info("temp save {} epcho train result, current best pass f1 {}".format(pass_id, pass_f1))
                fluid.io.save_persistables(dirname=train_parameters['save_model_dir'],
                                        main_program=train_program, executor=exe)
                current_best_f1 = pass_f1
                current_best_pass = pass_id
                current_best_recall = recall_category
                current_best_precision = precision_category

            logger.info("best pass {} current best pass f1 is {}".format(current_best_pass, current_best_f1))
            logger.info(
                "best pass {} current best pass recall_category is {}".format(current_best_pass, current_best_recall))
            logger.info(
                "best pass {} current best pass precision_category is {}".format(current_best_pass, current_best_precision))

        if pass_mean_loss < min_loss:
            logger.info("Has reached the set optimum value, the training is over")
            break

        if rise_count > rise_limit:
            logger.info("rise_count > rise_limit, so early stop")
            break
        else:
            if pass_mean_loss > train_temp_loss:
                rise_count += 1
                train_temp_loss = pass_mean_loss
            else:
                rise_count = 0
                train_temp_loss = pass_mean_loss

        if iters > max_iters:
            break

    logger.info("end training")


if __name__ == '__main__':
    train()
