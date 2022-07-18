import numpy as np
import pandas as pd
import os
import math
from tqdm import tqdm

import tensorflow as tf
from keras.layers import (Layer, Input, Reshape,Rescaling, Flatten, Dense, Dropout, TimeDistributed, Conv1D, 
                          Activation, LayerNormalization, Embedding, MultiHeadAttention, Lambda, GlobalMaxPooling1D,
                          MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, BatchNormalization, DepthwiseConv1D,
                          add, multiply, Concatenate, Add, ZeroPadding1D, SeparableConv1D)

from keras.models import Model
from keras import backend as K

def inc_res_conv1d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    
    x = Conv1D(filters,
                kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name)(x)

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 2
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis,
                                scale=False,
                                name=bn_name)(x)
                                
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):

    if block_type == 'block35':
        branch_0 = inc_res_conv1d_bn(x, 32, 1)
        branch_1 = inc_res_conv1d_bn(x, 32, 1)
        branch_1 = inc_res_conv1d_bn(branch_1, 32, 3)
        branch_2 = inc_res_conv1d_bn(x, 32, 1)
        branch_2 = inc_res_conv1d_bn(branch_2, 48, 3)
        branch_2 = inc_res_conv1d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = inc_res_conv1d_bn(x, 192, 1)
        branch_1 = inc_res_conv1d_bn(x, 128, 1)
        branch_1 = inc_res_conv1d_bn(branch_1, 160, 7)
        branch_1 = inc_res_conv1d_bn(branch_1, 192, 1)
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = inc_res_conv1d_bn(x, 192, 1)
        branch_1 = inc_res_conv1d_bn(x, 192, 1)
        branch_1 = inc_res_conv1d_bn(branch_1, 224, 3)
        branch_1 = inc_res_conv1d_bn(branch_1, 256, 1)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 2
    mixed = Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = inc_res_conv1d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x


def InceptionResNetV2(include_top=True,
                      weights=None,
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=100,
                      **kwargs):
    
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either `None` (random initialization) or the path to the weights file to be loaded.')

    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    # Stem block: 35 x 192
    x = inc_res_conv1d_bn(inputs, 32, 3, strides=2, padding='valid')
    x = inc_res_conv1d_bn(x, 32, 3, padding='valid')
    x = inc_res_conv1d_bn(x, 64, 3)
    x = MaxPooling1D(3, strides=2)(x)
    x = inc_res_conv1d_bn(x, 80, 1, padding='valid')
    x = inc_res_conv1d_bn(x, 192, 3, padding='valid')
    x = MaxPooling1D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 320
    branch_0 = inc_res_conv1d_bn(x, 96, 1)
    branch_1 = inc_res_conv1d_bn(x, 48, 1)
    branch_1 = inc_res_conv1d_bn(branch_1, 64, 5)
    branch_2 = inc_res_conv1d_bn(x, 64, 1)
    branch_2 = inc_res_conv1d_bn(branch_2, 96, 3)
    branch_2 = inc_res_conv1d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = inc_res_conv1d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 2
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 1088
    branch_0 = inc_res_conv1d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = inc_res_conv1d_bn(x, 256, 1)
    branch_1 = inc_res_conv1d_bn(branch_1, 256, 3)
    branch_1 = inc_res_conv1d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling1D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 2080
    branch_0 = inc_res_conv1d_bn(x, 256, 1)
    branch_0 = inc_res_conv1d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = inc_res_conv1d_bn(x, 256, 1)
    branch_1 = inc_res_conv1d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = inc_res_conv1d_bn(x, 256, 1)
    branch_2 = inc_res_conv1d_bn(branch_2, 288, 3)
    branch_2 = inc_res_conv1d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling1D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution blocks
    x = inc_res_conv1d_bn(x, 1024, 1, name='conv_7b')
    x = inc_res_conv1d_bn(x, 256, 1, name='conv_8b')
    x = inc_res_conv1d_bn(x, 64, 1, name='conv_9b')

    if include_top:
        # Classification block
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = None #utils.get_source_inputs(input_tensor)
    else:
        inputs = inputs

    # Create model.
    model = Model(inputs, x, name='Inception_ResNet_v2')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model



def Xception(include_top=False,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')


    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if K.image_data_format() == 'channels_first' else 2

    x = Conv1D(32, 3,
                      strides=2,
                      use_bias=False,
                      name='block1_conv1')(img_input)

    x = BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv1D(64, 3, use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv1D(128, 1,
                        strides=2,
                        padding='same',
                        use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = SeparableConv1D(128, 3,
                        padding='same',
                        use_bias=False,
                        name='block2_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv1D(128, 3,
                        padding='same',
                        use_bias=False,
                        name='block2_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

    x = MaxPooling1D(3,
                    strides=2,
                    padding='same',
                    name='block2_pool')(x)
    x = add([x, residual])

    residual = Conv1D(256, 1, 
                      strides=2,
                      padding='same',
                      use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv1D(256, 3,
                        padding='same',
                        use_bias=False,
                        name='block3_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv1D(256, 3,
                        padding='same',
                        use_bias=False,
                        name='block3_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

    x = MaxPooling1D(3, strides=2,
                            padding='same',
                            name='block3_pool')(x)
    x = add([x, residual])

    residual = Conv1D(728, 1,
                        strides=2,
                        padding='same',
                        use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv1D(728, 3,
                        padding='same',
                        use_bias=False,
                        name='block4_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv1D(728, 3,
                        padding='same',
                        use_bias=False,
                        name='block4_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

    x = MaxPooling1D(3, strides=2,
                            padding='same',
                            name='block4_pool')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv1D(728, 3,
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv1')(x)
        x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv1D(728, 3,
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv2')(x)
        x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv1D(728, 3,
                            padding='same',
                            use_bias=False,
                            name=prefix + '_sepconv3')(x)
        x = BatchNormalization(axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

        x = add([x, residual])

    residual = Conv1D(1024, 1, strides=2,
                      padding='same',
                      use_bias=False)(x)
    residual = BatchNormalization(axis=channel_axis)(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv1D(728, 3,
                        padding='same',
                        use_bias=False,
                        name='block13_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv1D(1024, 3,
                        padding='same',
                        use_bias=False,
                        name='block13_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)

    x = MaxPooling1D(3,
                    strides=2,
                    padding='same',
                    name='block13_pool')(x)
    x = add([x, residual])

    x = SeparableConv1D(256, 3,
                        padding='same',
                        use_bias=False,
                        name='block14_sepconv1')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv1D(64, 3,
                        padding='same',
                        use_bias=False,
                        name='block14_sepconv2')(x)
    x = BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = None #keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='Xception')

    # Load weights.
    if weights == 'imagenet':
        print("Can't find imagenet file for 1D xception")
    elif weights is not None:
        model.load_weights(weights)

    return model


def res_block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):

    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2

    preact = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_preact_bn')(x)
    preact = Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = Conv1D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = MaxPooling1D(1, strides=stride)(x) if stride > 1 else x

    x = Conv1D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    x = ZeroPadding1D(padding=(1, 1), name=name + '_2_pad')(x)
    x = Conv1D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv1D(4 * filters, 1, name=name + '_3_conv')(x)
    x = Add(name=name + '_out')([shortcut, x])
    return x


def res_stack2(x, filters, blocks, stride1=2, name=None):

    x = res_block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = res_block2(x, filters, name=name + '_block' + str(i))
    x = res_block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x

def res_conv1d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    
    x = Conv1D(filters,
                kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name)(x)

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 2
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis,
                                scale=False,
                                name=bn_name)(x)
                                
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           **kwargs):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2

    x = ZeroPadding1D(padding=(3,3) , name='conv1_pad')(img_input)
    x = Conv1D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)

    x = ZeroPadding1D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling1D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    # Final convolution blocks
    x = res_conv1d_bn(x, 1024, 1, name='conv_l1')
    x = res_conv1d_bn(x, 256, 1, name='conv_l2')
    x = res_conv1d_bn(x, 64, 1, name='conv_l3')

    if preact is True:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='post_bn')(x)
        x = Activation('relu', name='post_relu')(x)

    if include_top:
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = None #keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)

    # Load weights.
    if (weights == 'imagenet'):
        print("Imagenet weights not available for 1D ResNet")
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNet152V2(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    def stack_fn(x):
        x = res_stack2(x, 64, 3, name='conv2')
        x = res_stack2(x, 128, 8, name='conv3')
        x = res_stack2(x, 256, 36, name='conv4')
        x = res_stack2(x, 512, 3, stride1=1, name='conv5')
        return x
    return ResNet(stack_fn, False, True, 'ResNet152v2',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)



def dense_block(x, blocks, name):
    for i in range(blocks):
        x = densenet_conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def densenet_transition_block(x, reduction, name):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv1D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = AveragePooling1D(2, strides=2, name=name + '_pool')(x)
    return x


def densenet_conv_block(x, growth_rate, name):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv1D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv1D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def densenet_conv1d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    
    x = Conv1D(filters,
                kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name)(x)

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 2
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis,
                                scale=False,
                                name=bn_name)(x)
                                
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def DenseNet(blocks,
             include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2

    x = ZeroPadding1D(padding=(3, 3))(img_input)
    x = Conv1D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding1D(padding=(1, 1))(x)
    x = MaxPooling1D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = densenet_transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = densenet_transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = densenet_transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    # Final convolution blocks
    x = densenet_conv1d_bn(x, 640, 1, name=None)
    x = densenet_conv1d_bn(x, 256, 1, name=None)
    x = densenet_conv1d_bn(x, 64, 1, name=None)

    if include_top:
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = None #keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')

    # Load weights.
    if weights == 'imagenet':
        print("No weights for 1D DenseNet")
        
    elif weights is not None:
        model.load_weights(weights)

    return model

def DenseNet50(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([3, 6, 12, 8],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)

def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)
    

def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)
    
    
def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes,
                    **kwargs)


def resnext_block3(x, filters, kernel_size=3, stride=1, groups=32,
           conv_shortcut=True, name=None):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2

    if conv_shortcut is True:
        shortcut = Conv1D((64 // groups) * filters, 1, strides=stride,
                                 use_bias=False, name=name + '_0_conv')(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = Conv1D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = ZeroPadding1D(padding=(1, 1), name=name + '_2_pad')(x)
    x = DepthwiseConv1D(kernel_size, strides=stride, depth_multiplier=c,
                               use_bias=False, name=name + '_2_conv')(x)
    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)

    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.

    x = Conv1D(filters, 1, use_bias=False, trainable=False,
                      kernel_initializer={'class_name': 'Constant',
                                          'config': {'value': kernel}},
                      name=name + '_2_gconv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_relu')(x)

    x = Conv1D((64 // groups) * filters, 1,
                      use_bias=False, name=name + '_3_conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = Add(name=name + '_add')([shortcut, x])
    x = Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    x = resnext_block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = resnext_block3(x, filters, groups=groups, conv_shortcut=False,
                   name=name + '_block' + str(i))
    return x

def resnext_conv1d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    
    x = Conv1D(filters,
                kernel_size,
                strides=strides,
                padding=padding,
                use_bias=use_bias,
                name=name)(x)

    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 2
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis,
                                scale=False,
                                name=bn_name)(x)
                                
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def ResNet_2(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           **kwargs):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2

    x = ZeroPadding1D(padding=(3,3) , name='conv1_pad')(img_input)
    x = Conv1D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='conv1_bn')(x)
        x = Activation('relu', name='conv1_relu')(x)

    x = ZeroPadding1D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling1D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    # Final convolution blocks
    x = resnext_conv1d_bn(x, 1024, 1, name=None)
    x = resnext_conv1d_bn(x, 256, 1, name=None)
    x = resnext_conv1d_bn(x, 64, 1, name=None)

    if preact is True:
        x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='post_bn')(x)
        x = Activation('relu', name='post_relu')(x)

    if include_top:
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = None #keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)

    # Load weights.
    if (weights == 'imagenet'):
        print("Imagenet weights not available for 1D ResNext")
    elif weights is not None:
        model.load_weights(weights)

    return model


def ResNeXt101(include_top=True,
               weights='imagenet',
               input_tensor=None,
               input_shape=None,
               pooling=None,
               classes=1000,
               **kwargs):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 23, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x
    return ResNet_2(stack_fn, False, False, 'ResNeXt101',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)

def ResNeXt50(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              **kwargs):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 6, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x
    return ResNet_2(stack_fn, False, False, 'ResNeXt50',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)        



EFF_DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

EFF_CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

EFF_DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def eff_correct_pad(K, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 1
    input_size = K.int_shape(inputs)[img_dim]

    if input_size is None:
        adjust = 1
    else:
        adjust = 1 - input_size % 2

    correct = kernel_size // 2

    return (correct - adjust, correct)
    
def eff_swish(x):
    """eff_swish activation function.
    # Arguments
        x: Input tensor.
    # Returns
        The eff_swish activation: `x * sigmoid(x)`.
    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if K.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return K.tf.nn.eff_swish(x)
        except AttributeError:
            pass

    return x * K.sigmoid(x)


def eff_block(inputs, activation_fn=eff_swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = Conv1D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=EFF_CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = ZeroPadding1D(padding=eff_correct_pad(K, x, kernel_size), 
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = DepthwiseConv1D(kernel_size,    
                        strides=strides, 
                        padding=conv_pad, 
                        use_bias=False,
                        depthwise_initializer=EFF_CONV_KERNEL_INITIALIZER, 
                        name=name + 'dwconv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = GlobalAveragePooling1D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = Reshape((filters, 1), name=name + 'se_reshape')(se)
        else:
            se = Reshape((1, filters), name=name + 'se_reshape')(se)
        se = Conv1D(filters_se, 1,
                           padding='same',
                           activation=activation_fn,
                           kernel_initializer=EFF_CONV_KERNEL_INITIALIZER,
                           name=name + 'se_reduce')(se)
        se = Conv1D(filters, 1,
                           padding='same',
                           activation='sigmoid',
                           kernel_initializer=EFF_CONV_KERNEL_INITIALIZER,
                           name=name + 'se_expand')(se)
        x = multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = Conv1D(filters_out, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=EFF_CONV_KERNEL_INITIALIZER,
                      name=name + 'project_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = Dropout(drop_rate,
                               noise_shape=(None, 1, 1) ,
                               name=name + 'drop')(x)
        x = add([x, inputs], name=name + 'add')

    return x


def EfficientNet(width_coefficient,
                 depth_coefficient,
                 default_size,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 depth_divisor=8,
                 activation_fn=eff_swish,
                 blocks_args=EFF_DEFAULT_BLOCKS_ARGS,
                 model_name='efficientnet',
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000,
                 **kwargs):


    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 1 if K.image_data_format() == 'channels_first' else 2

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    x = ZeroPadding1D(padding=eff_correct_pad(K, x, 3),
                             name='stem_conv_pad')(x)
    x = Conv1D(round_filters(32), 3,
                      strides=2,
                      padding='valid',
                      use_bias=False,
                      kernel_initializer=EFF_CONV_KERNEL_INITIALIZER,
                      name='stem_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = Activation(activation_fn, name='stem_activation')(x)

    # Build blocks
    from copy import deepcopy
    blocks_args = deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = eff_block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1

    # Build top

    x = Conv1D(144, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=EFF_CONV_KERNEL_INITIALIZER,
                      name=None)(x)
    x = BatchNormalization(axis=bn_axis, name=None)(x)
    x = Activation(activation_fn, name=None)(x)

    x = Conv1D(64, 1,
                padding='same',
                use_bias=False,
                kernel_initializer=EFF_CONV_KERNEL_INITIALIZER,
                name=None)(x)
    x = BatchNormalization(axis=bn_axis, name=None)(x)
    x = Activation(activation_fn, name=None)(x)

    if include_top:
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name='top_dropout')(x)
        x = Dense(classes,
                         activation='softmax',
                         kernel_initializer=EFF_DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling1D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling1D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = None #keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
        print("ImageNet weights not available for 1D EfficientNet")
    elif weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetB0(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.0, 224, 0.2,
                        model_name='efficientnet-b0',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB1(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.0, 1.1, 240, 0.2,
                        model_name='efficientnet-b1',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB2(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.1, 1.2, 260, 0.3,
                        model_name='efficientnet-b2',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB3(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.2, 1.4, 300, 0.3,
                        model_name='efficientnet-b3',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB4(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.4, 1.8, 380, 0.4,
                        model_name='efficientnet-b4',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB5(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.6, 2.2, 456, 0.4,
                        model_name='efficientnet-b5',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB6(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(1.8, 2.6, 528, 0.5,
                        model_name='efficientnet-b6',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)


def EfficientNetB7(include_top=True,
                   weights=None,
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000,
                   **kwargs):
    return EfficientNet(2.0, 3.1, 600, 0.5,
                        model_name='efficientnet-b7',
                        include_top=include_top, weights=weights,
                        input_tensor=input_tensor, input_shape=input_shape,
                        pooling=pooling, classes=classes,
                        **kwargs)

