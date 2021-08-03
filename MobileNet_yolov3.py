# -*- coding: UTF-8 -*-
"""
主干网络基于mobilenetv1的yolo3
"""
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA


class YOLOv3(object):
    """
    定义模型
    """

    def __init__(self, class_num, anchors, anchor_mask, scale=1.0, is_train=True):
        """
        初始化模型参数
        :param class_num:
        :param anchors:
        :param anchor_mask:
        :param scale:
        :param is_train:
        """
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num
        self.scale = scale
        self.is_train = is_train
        self.yolo_anchors = []
        self.yolo_classes = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        """
        name
        :return:
        """
        return 'YOLOv3'

    def get_anchors(self):
        """
        get_anchors
        :return:
        """
        return self.anchors

    def get_anchor_mask(self):
        """
        get_anchor_mask
        :return:
        """
        return self.anchor_mask

    def get_class_num(self):
        """
        get_class_num
        :return:
        """
        return self.class_num

    def get_downsample_ratio(self):
        """
        get_downsample_ratio
        :return:
        """
        return self.downsample_ratio

    def get_yolo_anchors(self):
        """
        get_yolo_anchors
        :return:
        """
        return self.yolo_anchors

    def get_yolo_classes(self):
        """
        get_yolo_classes
        :return:
        """
        return self.yolo_classes

    def conv_bn_(self,
                 input,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1,
                 act='relu',
                 use_cudnn=True):
        """
        mobilenetv1中的conv+bn
        :param input:
        :param filter_size:
        :param num_filters:
        :param stride:
        :param padding:
        :param num_groups:
        :param act:
        :param use_cudnn:
        :return:
        """
        parameter_attr = ParamAttr(learning_rate=0.1, initializer=MSRA())
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)

    def depthwise_separable(self, input, num_filters1, num_filters2, num_groups,
                            stride, scale):
        """
        DW卷积
        :param input:
        :param num_filters1:
        :param num_filters2:
        :param num_groups:
        :param stride:
        :param scale:
        :return:
        """
        depthwise_conv = self.conv_bn_(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=True)

        pointwise_conv = self.conv_bn_(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)
        return pointwise_conv

    def conv_bn_layer(self, input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='leaky',
                      is_test=True,
                      name=None):
        """
        detection block中的conv+bn
        :param input:
        :param ch_out:
        :param filter_size:
        :param stride:
        :param padding:
        :param act:
        :param is_test:
        :param name:
        :return:
        """
        conv1 = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        out = fluid.layers.batch_norm(
            input=conv1,
            act=None,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.),
                name=bn_name + '.scale'),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.),
                name=bn_name + '.offset'),
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')
        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

    def yolo_detection_block(self, input, channel, is_test=True, name=None):
        """
        detection block
        :param input:
        :param channel:
        :param is_test:
        :param name:
        :return:
        """

        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)
        conv = input
        for j in range(2):
            conv = self.conv_bn_layer(
                conv,
                channel,
                filter_size=1,
                stride=1,
                padding=0,
                is_test=is_test,
                name='{}.{}.0'.format(name, j))
            conv = self.conv_bn_layer(
                conv,
                channel * 2,
                filter_size=3,
                stride=1,
                padding=1,
                is_test=is_test,
                name='{}.{}.1'.format(name, j))
        route = self.conv_bn_layer(
            conv,
            channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
            name='{}.2'.format(name))
        tip = self.conv_bn_layer(
            route,
            channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
            name='{}.tip'.format(name))
        return route, tip

    def upsample(self, input, scale=2, name=None):
        """
        up sample
        :param input:
        :param scale:
        :param name:
        :return:
        """
        out = fluid.layers.resize_nearest(
            input=input, scale=float(scale), name=name)
        return out

    def net(self, input):
        """
        模型构建
        :param input:
        :return:
        """

        scale = self.scale
        blocks = []
        tmp = self.conv_bn_(input, 3, int(32 * scale), 2, 1)
        print(tmp)
        self.downsample_ratio *= 2
        blocks.append(tmp)
        tmp = self.depthwise_separable(tmp, 32, 64, 32, 1, scale)
        tmp = self.depthwise_separable(tmp, 64, 128, 64, 2, scale)
        self.downsample_ratio *= 2
        blocks.append(tmp)
        tmp = self.depthwise_separable(tmp, 128, 128, 128, 1, scale)
        tmp = self.depthwise_separable(tmp, 128, 256, 128, 2, scale)
        self.downsample_ratio *= 2
        blocks.append(tmp)
        tmp = self.depthwise_separable(tmp, 256, 256, 256, 1, scale)
        tmp = self.depthwise_separable(tmp, 256, 512, 256, 2, scale)
        self.downsample_ratio *= 2
        blocks.append(tmp)
        for i in range(5):
            tmp = self.depthwise_separable(tmp, 512, 512, 512, 1, scale)

        tmp = self.depthwise_separable(tmp, 512, 1024, 512, 2, scale)
        tmp = self.depthwise_separable(tmp, 1024, 1024, 1024, 1, scale)
        self.downsample_ratio *= 2
        blocks.append(tmp)
        print(len(blocks))
        for i in range(len(blocks)):
            print(blocks[i].shape)

        blocks = [blocks[-1], blocks[-2], blocks[-3]]
        print(blocks)

        # yolo detector
        for i, block in enumerate(blocks):
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self.yolo_detection_block(
                block,
                channel=512 // (2 ** i),
                is_test=not self.is_train,
                name="yolo_block.{}".format(i))

            # out channel number = mask_num * (5 + class_num)
            num_filters = len(self.anchor_mask[i]) * (self.class_num + 5)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=num_filters,
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(
                    initializer=fluid.initializer.Normal(0., 0.02),
                    name="yolo_output.{}.conv.weights".format(i)),
                bias_attr=ParamAttr(
                    initializer=fluid.initializer.Constant(0.0),
                    regularizer=L2Decay(0.),
                    name="yolo_output.{}.conv.bias".format(i)))
            self.outputs.append(block_out)

            if i < len(blocks) - 1:
                route = self.conv_bn_layer(
                    input=route,
                    ch_out=256 // (2 ** i),
                    filter_size=1,
                    stride=1,
                    padding=0,
                    is_test=(not self.is_train),
                    name="yolo_transition.{}".format(i))
                # upsample
                route = self.upsample(route)

        return self.outputs


class YOLOv3Tiny(object):
    """
    yolo_tiny
    """

    def __init__(self, class_num, anchors, anchor_mask):
        """
        初始化模型的超参数
        """
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors
        self.class_num = class_num

        self.yolo_anchors = []
        self.yolo_classes = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)
            self.yolo_classes.append(class_num)

    def name(self):
        """
        name
        :return:
        """
        return 'YOLOv3'

    def get_anchors(self):
        """
        get_anchors
        :return:
        """
        return self.anchors

    def get_anchor_mask(self):
        """
        get_anchor_mask
        :return:
        """
        return self.anchor_mask

    def get_class_num(self):
        """
        get_class_num
        :return:
        """
        return self.class_num

    def get_downsample_ratio(self):
        """
        get_downsample_ratio
        :return:
        """
        return self.downsample_ratio

    def get_yolo_anchors(self):
        """
        get_yolo_anchors
        :return:
        """
        return self.yolo_anchors

    def get_yolo_classes(self):
        """
        get_yolo_classes
        :return:
        """
        return self.yolo_classes

    def conv_bn(self,
                input,
                num_filters,
                filter_size,
                stride,
                padding,
                num_groups=1,
                use_cudnn=True):
        """
        卷积 + bn
        :return:
        """
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            groups=num_groups,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False)

        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        out = fluid.layers.batch_norm(
            input=conv, act='relu',
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))

        return out

    def depthwise_conv_bn(self, input, filter_size=3, stride=1, padding=1):
        """
        深度可分离卷积 + bn
        :return:
        """
        num_filters = input.shape[1]
        return self.conv_bn(input,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            num_groups=num_filters)

    def downsample(self, input, pool_size=2, pool_stride=2):
        """
        通过池化进行下采样
        :return:
        """
        self.downsample_ratio *= 2
        return fluid.layers.pool2d(input=input, pool_type='max', pool_size=pool_size,
                                   pool_stride=pool_stride)

    def basicblock(self, input, num_filters):
        """
        基础的卷积块
        :return:
        """
        conv1 = self.conv_bn(input, num_filters, filter_size=3, stride=1, padding=1)
        out = self.downsample(conv1)
        return out

    def upsample(self, input, scale=2):
        """
        上采样
        :return:
        """
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(input)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=input,
            scale=scale,
            actual_shape=out_shape)
        return out

    def yolo_detection_block(self, input, num_filters):
        """
        yolo检测模块
        :return:
        """
        route = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    def net(self, img):
        """
        整体网络结构构建
        :return:
        """
        # darknet-tiny
        stages = [16, 32, 64, 128, 256, 512]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than downsample times"
        # 256x256
        tmp = img
        blocks = []
        for i, stage_count in enumerate(stages):
            if i == len(stages) - 1:
                block = self.conv_bn(tmp, stage_count, filter_size=3, stride=1, padding=1)
                blocks.append(block)
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.conv_bn(blocks[-1], stage_count * 2, filter_size=1, stride=1, padding=0)
                blocks.append(block)
            else:
                tmp = self.basicblock(tmp, stage_count)
                blocks.append(tmp)

        blocks = [blocks[-1], blocks[3]]

        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            if i < 1:
                route, tip = self.yolo_detection_block(block, num_filters=256 // (2 ** i))
            else:
                tip = self.conv_bn(block, num_filters=256, filter_size=3, stride=1, padding=1)
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * (self.class_num + 5),  # 5 elements represent x|y|h|w|score
                filter_size=1,
                stride=1,
                padding=0,
                act=None,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 128 // (2 ** i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs


def get_yolo(is_tiny, class_num, anchors, anchor_mask):
    """
    根据is_tiny来构建yolo网络或yolo_tiny
    :return:
    """
    if is_tiny:
        return YOLOv3Tiny(class_num, anchors, anchor_mask)
    else:
        return YOLOv3(class_num, anchors, anchor_mask)
