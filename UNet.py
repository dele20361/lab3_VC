# Universidad del Valle de Guatemala 
# Vision por computador

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, concatenate, Conv2D, Dropout, Input

class UNet():
    def down(self, x, n_filters, drop_p=0, name=''):
        x = Conv2D(n_filters, 3, activation='relu', padding='same', name=name+'conv1_down')(x)
        x = Conv2D(n_filters, 3, activation='relu', padding='same', name=name+'conv2_down')(x)
        x = MaxPooling2D(pool_size=(2,2), name=name+'pool_down')(x)
        x = Dropout(drop_p, name=name+'_dropoutdown')(x)
        return x

    def up(self, x, conv_features, n_filters, drop_p=0, name=''):
        x = UpSampling2D(size=(2, 2), name=name+'upsampling_up')(x)
        x = concatenate([x, conv_features], axis=3, name=name+'concat_up')
        x = Conv2D(n_filters, 3, activation='relu', padding='same', name=name+'conv1_up')(x)
        x = Conv2D(n_filters, 3, activation='relu', padding='same', name=name+'conv2_up')(x)
        x = Dropout(drop_p, name=name+'_dropoutup')(x)
        return x

    def build_model(self, input_shape, levels=4, filters=16, drop_p=0.1):
        inputs = Input(shape=input_shape, name='input')

        # Downsampling Path
        x = inputs
        skips = []
        for level in range(levels):
            x = self.down(x, filters * 2**level, drop_p=drop_p, name=f'down_{level}')
            skips.append(x)

        # Upsampling Path
        for level in reversed(range(levels-1)):
            x = self.up(x, skips[level], filters * 2**level, drop_p=drop_p, name=f'up_{level}')

        # Output layer
        outputs = Conv2D(3, (1, 1), activation='relu', padding='same', name='output')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='u-net')

        return model
