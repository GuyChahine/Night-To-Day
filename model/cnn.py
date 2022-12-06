import numpy as np

from keras.models import Model
from keras import layers
from keras.optimizers import Adam

import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("Night-To-Day")+1]))

from src.data_generator import DataGenerator

class CNN():
    def __init__(
        self,
        image_shape = (64,64,3),
        batch_size = 50,
        res_filters = 64,
        conv_filters = 16,
        nb_resnet = 3,
        optimizer = Adam(),
        data_generator = DataGenerator((64,64), 50)
    ):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.res_filters = res_filters
        self.conv_filters = conv_filters
        self.nb_resnet = nb_resnet
        self.optimizer = optimizer
        self.data_generator = data_generator
        
        self.model = self.combined()
        self.model.compile(self.optimizer, 'mse', metrics=['accuracy'])
    
    def resnet_block(self, input_image):
        x = layers.Conv2D(self.res_filters, (3,3), padding='same')(input_image)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(self.res_filters, (3,3), padding='same')(x)
        
        x = layers.Concatenate()([x, input_image])
        return x
    
    def combined(self):
        
        def conv2d(input_layer, filters, kernel, strides):
            x = layers.Conv2D(filters, kernel, strides=strides, padding='same')(input_layer)
            x = layers.Activation('relu')(x)
            return x
        
        def conv2dt(input_layer, filters, kernel, strides):
            x = layers.Conv2DTranspose(filters, kernel, strides=strides, padding='same')(input_layer)
            x = layers.Activation('relu')(x)
            return x
        
        input_image = layers.Input(shape=self.image_shape)

        x = conv2d(input_image, self.conv_filters*1, (7,7), (1,1))
        x = conv2d(x, self.conv_filters*2, (3,3), (2,2))
        x = conv2d(x, self.conv_filters*4, (3,3), (2,2))
        
        for _ in range(self.nb_resnet):
            x = self.resnet_block(x)
        
        x = conv2dt(x, self.conv_filters*2, (3,3), (2,2))
        x = conv2dt(x, self.conv_filters*1, (3,3), (2,2))
        
        x = layers.Conv2D(3, (7,7), padding='same')(x)
        output_image = layers.Activation('tanh')(x)
        
        return Model(input_image, output_image)
    
    def train(self, epochs):
        for epoch in range(epochs):
            for batch_i, (image_night, image_day) in enumerate(self.data_generator):
                model_loss = self.model.train_on_batch(image_night, image_day)
                
                print("[Epoch {}/{}] [Batch {}/{}] [Loss {:.5f}] [Accuracy {:.5f}]".format(
                    epoch+1, epochs,
                    batch_i+1, self.data_generator.nb_batch,
                    model_loss[0], model_loss[1]
                ))