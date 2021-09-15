"""A vanilla 3D resnet implementation.

Based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

import keras.layers
import six
from math import ceil
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Conv3D,
    Conv3DTranspose,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(is_decoder,**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))
    if is_decoder == False:
        def f(input):
            conv = Conv3D(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding,
                          kernel_regularizer=kernel_regularizer)(input)
            return _bn_relu(conv)
    else:
        def f(input):
            conv = Conv3DTranspose(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding,
                          kernel_regularizer=kernel_regularizer)(input)
            return _bn_relu(conv)

    return f


def _bn_relu_conv3d(is_decoder,**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))
    if is_decoder==False:
        def f(input):
            activation = _bn_relu(input)
            return Conv3D(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding,
                          kernel_regularizer=kernel_regularizer)(activation)
    if is_decoder==True:
        def f(input):
            activation = _bn_relu(input)
            return Conv3DTranspose(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding,
                          kernel_regularizer=kernel_regularizer)(activation)
    return f


def _shortcut3d(input, residual,is_decoder=False):
    """3D shortcut to match input and residual and merges them with "sum"."""#ceil(input.shape[DIM1_AXIS] \ / residual._keras_shape[DIM1_AXIS])
    if is_decoder==False:
        stride_dim1 = ceil(input.shape[DIM1_AXIS] \
            / residual.shape[DIM1_AXIS])
        stride_dim2 = ceil(input.shape[DIM2_AXIS] \
            / residual.shape[DIM2_AXIS])
        stride_dim3 = ceil(input.shape[DIM3_AXIS] \
            / residual.shape[DIM3_AXIS])
        equal_channels = residual.shape[CHANNEL_AXIS] \
            == input.shape[CHANNEL_AXIS]
    elif is_decoder==True:
        stride_dim1 = ceil(residual.shape[DIM1_AXIS] \
                           / input.shape[DIM1_AXIS])
        stride_dim2 = ceil(residual.shape[DIM2_AXIS] \
                           / input.shape[DIM2_AXIS])
        stride_dim3 = ceil(residual.shape[DIM3_AXIS] \
                           / input.shape[DIM3_AXIS])
        equal_channels = residual.shape[CHANNEL_AXIS] \
                         == input.shape[CHANNEL_AXIS]
    #print('shape:',is_decoder, stride_dim1, stride_dim2, stride_dim3)
    #print('input:',input)
    #print('input:', residual)
    shortcut = input
    if is_decoder==False:
        if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
                or not equal_channels:
            shortcut = Conv3D(
                filters=residual.shape[CHANNEL_AXIS],
                kernel_size=(1, 1, 1),
                strides=(stride_dim1, stride_dim2, stride_dim3),
                kernel_initializer="he_normal", padding="valid",
                kernel_regularizer=l2(1e-4)
                )(input)
    elif is_decoder==True:
        if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
                or not equal_channels:
            shortcut = Conv3DTranspose(
                filters=residual.shape[CHANNEL_AXIS],
                kernel_size=(1, 1, 1),
                strides=(stride_dim1, stride_dim2, stride_dim3),
                kernel_initializer="he_normal", padding="valid",
                kernel_regularizer=l2(1e-4)
                )(input)
    return add([shortcut, residual])


def _residual_block3d(block_function, filters, kernel_regularizer, repetitions,
                      is_first_layer=False,is_bridege=False,is_decoder=True):#加is_bridege=False限制strides,encoder的stride=(2,2,2),bridge的stride=(1,1,1)
    if is_decoder==False:
        def f(input):
            for i in range(repetitions):
                strides = (1, 1, 1)
                if not is_bridege:
                    strides = (2, 2, 2)
                input = block_function(filters=filters, strides=strides,
                                       kernel_regularizer=kernel_regularizer,
                                       is_first_block_of_first_layer=(
                                           is_first_layer and i == 0),is_decoder=False)(input)
                #print('input1_res:',is_bridege,input)
            return input
    if is_decoder == True:
        def f(input):
            for i in range(repetitions):
                strides = (1, 1, 1)
                if not is_bridege:
                    strides = (2, 2, 2)
                input = block_function(filters=filters, strides=strides,
                                       kernel_regularizer=kernel_regularizer,
                                       is_first_block_of_first_layer=(
                                           is_first_layer and i == 0),is_decoder=True)(input)
                #print('input2:', input)
            return input

    return f


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer
                           )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv1)
        return _shortcut3d(input, residual)

    return f


def bottleneck(filters, strides=(2, 2, 2), kernel_regularizer=l2(1e-4),
               is_first_block_of_first_layer=False,is_decoder=False,is_bridege=True):

    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    if is_bridege==True:
        a=1
    elif is_bridege==False:
        a=2
    if is_decoder==False:
        def f(input):
            #if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                                  strides=strides, padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=kernel_regularizer
                                  )(input)
            '''else:
                conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),#_bn_relu_conv3d的strides已经设定为1
                                           strides=strides,
                                           kernel_regularizer=kernel_regularizer
                                           )(input)'''

            conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                       kernel_regularizer=kernel_regularizer,
                                       is_decoder = False
                                       )(conv_1_1)
            residual = _bn_relu_conv3d(filters=filters * a, kernel_size=(1, 1, 1),
                                       kernel_regularizer=kernel_regularizer,
                                       is_decoder = False
                                       )(conv_3_3)
            #print('input1:',a, input)
            #print('residual:', residual)
            return _shortcut3d(input, residual,is_decoder=False)
    if is_decoder == True:
        def f(input):
            #if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3DTranspose(filters=filters, kernel_size=(1, 1, 1),
                                  strides=strides, padding="same",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=kernel_regularizer,
                                  )(input)
            '''else:
                conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                           strides=strides,
                                           kernel_regularizer=kernel_regularizer,
                                           is_decoder=True)(input)'''
            conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                       kernel_regularizer=kernel_regularizer,
                                       is_decoder=True)(conv_1_1)
            residual = _bn_relu_conv3d(filters=filters /2, kernel_size=(1, 1, 1),
                                       kernel_regularizer=kernel_regularizer,
                                       is_decoder=True)(conv_3_3)
            #print('input1:', a, input)
            #print('residual:', residual)
            #return _shortcut3d(input, residual,is_decoder==True)
            return residual

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class Resnet3DBuilder1(object):
    """ResNet3D."""

    @staticmethod
    def build(input_shape, block_fn, repetitions1,repetitions2,repetitions3, reg_factor):
        """Instantiate a vanilla ResNet3D keras model.

        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        block_fn = _get_block(block_fn)
        input = Input(shape=input_shape)
        # first conv
        conv1 = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
                                strides=(2, 2, 2),
                                kernel_regularizer=l2(reg_factor),
                                is_decoder=False
                                )(input)#(45,32,32,64)
        #encoder
        filters=64
        for i, r in enumerate(repetitions1):
            conv1 = _residual_block3d(block_fn, filters=filters,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0),
                                      is_bridege=False,is_decoder=False)(conv1)
            filters *= 2

        block =_bn_relu(conv1)#()
        # repeat blocks
        wave=_conv_bn_relu3D(filters=3, kernel_size=(3, 3, 3),
                                strides=(1, 4, 4),
                                kernel_regularizer=l2(reg_factor),
                                is_decoder=False
                                )(block)#(4,1,1,3)
        wave=_conv_bn_relu3D(filters=3, kernel_size=(3, 3, 3),
                            strides=(16, 1, 1),
                            kernel_regularizer=l2(reg_factor),
                            is_decoder=True
                            )(wave)  # (64,1,1,3)

        filters1 = 512
        for i, r in enumerate(repetitions2):
            block = _residual_block3d(block_fn, filters=filters1,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0),
                                      is_bridege=True,is_decoder=False)(block)
            filters1 =filters1
        # last activation
        block_output = _bn_relu(block)
        block_output = 40 * block_output  # 放大系数,训练用25
        #decoder
        filters2 = 512
        for i, r in enumerate(repetitions3):
            block_output = _residual_block3d(block_fn, filters=filters2,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0),
                                      is_bridege=False,is_decoder=True)(block_output)
            filters2 /= 2
            block_output = _conv_bn_relu3D(filters=filters2, kernel_size=(3, 3, 3),
                                   strides=(1, 1, 1),
                                   kernel_regularizer=l2(reg_factor),
                                   is_decoder=True
                                   )(block_output)
            #print('block:', i, r, filters, conv1)
        # last activation
        block_output = _bn_relu(block_output)

        out=_conv_bn_relu3D(filters=3, kernel_size=(7, 7, 7),
                                strides=(2, 2, 2),
                                kernel_regularizer=l2(reg_factor),
                                is_decoder=True
                                )(block_output)
        #out=out*wave
        wave = Flatten()(wave)
        out_img=add([out,input])
        model = Model(inputs=input, outputs=[out,out_img])
        return model

    @staticmethod
    def encoder_decoder(input_shape,  reg_factor=1e-4):
        return Resnet3DBuilder1.build(input_shape, bottleneck,
                                     repetitions1=[1,1,1],repetitions2=[2],repetitions3=[1,1,1],reg_factor=reg_factor)

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 18."""
        return Resnet3DBuilder1.build(input_shape, num_outputs, basic_block,
                                     [2, 2, 2, 2], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 34."""
        return Resnet3DBuilder1.build(input_shape, num_outputs, basic_block,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 50."""
        return Resnet3DBuilder1.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 101."""
        return Resnet3DBuilder1.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 23, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 152."""
        return Resnet3DBuilder1.build(input_shape, num_outputs, bottleneck,
                                     [3, 8, 36, 3], reg_factor=reg_factor)



