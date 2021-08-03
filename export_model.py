# -*- coding: UTF-8 -*-
"""
模型固化
"""
import paddle.fluid as fluid
from MobileNet_yolov3 import get_yolo
import config
from paddleslim.prune import Pruner
from paddleslim.analysis import flops
import paddle 
import os
paddle.enable_static()
train_parameters = config.init_train_parameters()


def freeze_model(score_threshold):
    """
    模型固化
    :param score_threshold:
    :return:
    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    ues_tiny = train_parameters['use_tiny']
    yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']
    path = train_parameters['save_model_dir']
    path1 = train_parameters['pretrained_model_dir']
    model = get_yolo(ues_tiny, train_parameters['class_dim'], yolo_config['anchors'], yolo_config['anchor_mask'])
    image = fluid.layers.data(name='image', shape=yolo_config['input_size'], dtype='float32')
    image_shape = fluid.layers.data(name="image_shape", shape=[2], dtype='float32')

    boxes = []
    scores = []
    outputs = model.net(image)
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
        score_threshold=score_threshold,
        nms_top_k=train_parameters['nms_top_k'],
        keep_top_k=train_parameters['nms_pos_k'],
        nms_threshold=train_parameters['nms_thresh'],
        background_label=-1,
        name="multiclass_nms")

    #print(pred)

    scope = fluid.global_scope()
    startup_prog = fluid.default_startup_program()
    freeze_program = fluid.default_main_program()
    exe.run(startup_prog, scope=scope)
    #fluid.io.load_persistables(exe, path, freeze_program)
    def if_exist(var):
            """
            if_exist
            :param var:
            :return:
            """
            return os.path.exists(os.path.join(train_parameters['pretrained_model_dir'], var.name))

    fluid.io.load_vars(exe, train_parameters['pretrained_model_dir'], main_program=freeze_program,predicate=if_exist)
    freeze_program = freeze_program.clone(for_test=True)

    pruned_params = train_parameters['pruned_params'].strip().split(",")
    # logger.info("pruned params: {}".format(pruned_params))
    pruned_ratios = [float(n) for n in train_parameters['pruned_ratios'].strip().split(",")]
    # logger.info("pruned ratios: {}".format(pruned_ratios))

    base_flops = flops(freeze_program)
    pruner = Pruner()
    freeze_program, _, _ = pruner.prune(
        freeze_program,
        fluid.global_scope(),
        params=pruned_params,
        ratios=pruned_ratios,
        place=place,
        only_graph=False)
    pruned_flops = flops(freeze_program)

    print("pruned FLOPS: {}".format(
        float(base_flops - pruned_flops) / base_flops))

    exe.run(startup_prog)
    fluid.io.load_persistables(exe, path, freeze_program)
    # print("freeze out: {0}, pred layout: {1}".format(train_parameters['freeze_dir'], pred))
    fluid.io.save_inference_model(train_parameters['freeze_dir'], ['image', 'image_shape'], pred, exe, freeze_program,
                                  model_filename='__model__', params_filename='__params__')
    # fluid.io.save_inference_model(train_parameters['freeze_dir'], ['image'], outputs, exe, freeze_program)
    print("freeze end")


if __name__ == '__main__':
    freeze_model(0.01)
