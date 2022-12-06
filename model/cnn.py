import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Model
from keras import layers
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.initializers import RandomNormal

import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("Night-To-Day")+1]))

from src.data_generator import DataGenerator

class Utils():
    def train(self, epochs):
        continue_training = 0 if len(self.epoch) == 0 else self.epoch[-1]
        for epoch in range(epochs):
            for batch_i, (image_night, image_day) in enumerate(self.data_generator):
                model_loss = self.model.train_on_batch(image_night, image_day)
                
                print("[Epoch {}/{}] [Batch {}/{}] [Loss {:.5f}] [Accuracy {:.5f}]".format(
                    epoch+1, epochs,
                    batch_i+1, self.data_generator.nb_batch,
                    model_loss[0], model_loss[1]
                ))
                
                self.epoch.append( continue_training + epoch + ( (batch_i+1)/self.data_generator.nb_batch ) )
                self.loss.append(model_loss[0])
                self.acc.append(model_loss[1])
            if ( (epoch+1) % self.epoch_save_interval) == 0:
                self.export_sample(self.run_name)
        
    
    def export_sample(self, name):
        plt.rcParams["figure.autolayout"] = True
        
        night_image, day_image = self.sample_generator.__getitem__(0)
        predicted = self.model.predict(night_image)
        plt.figure(figsize=(20,8))
        plt.subplot(1,3,1)
        plt.title("DAY IMAGE")
        plt.imshow(day_image[0])
        plt.subplot(1,3,2)
        plt.title("NIGHT IMAGE")
        plt.imshow(night_image[0])
        plt.subplot(1,3,3)
        plt.title("PREDICTED IMAGE")
        plt.imshow(predicted[0])
        plt.savefig(f"../resources/sample/{name}_e{self.epoch[-1]}.jpg")
        
    def export_report(self, name):
        pd.DataFrame({
            "epoch": self.epoch,
            "loss": self.loss,
            "acc": self.acc
        }).to_csv(f"../resources/reports/{name}.csv")

class CNN(Utils):
    def __init__(
        self,
        run_name,
        image_shape = (64,64,3),
        batch_size = 50,
        res_filters = 256,
        conv_filters = 64,
        nb_resnet = 9,
        optimizer = Adam(),
        data_generator = DataGenerator((64,64), 50),
    ):
        self.run_name = run_name
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.res_filters = res_filters
        self.conv_filters = conv_filters
        self.nb_resnet = nb_resnet
        self.optimizer = optimizer
        self.data_generator = data_generator
        
        self.sample_generator = DataGenerator(self.image_shape[:2], 1)
        self.epoch, self.loss, self.acc = [], [], []
        
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


class CNN2(Utils):
    def __init__(
        self,
        run_name,
        image_shape = (64,64,3),
        batch_size = 20,
        res_filters = 256,
        conv_filters = 64,
        nb_resnet = 18,
        optimizer = Adam(),
        data_generator = DataGenerator((64,64), 20),
        epoch_save_interval = 20
    ):
        self.run_name = run_name
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.res_filters = res_filters
        self.conv_filters = conv_filters
        self.nb_resnet = nb_resnet
        self.optimizer = optimizer
        self.data_generator = data_generator
        self.epoch_save_interval = epoch_save_interval
        
        self.sample_generator = DataGenerator(self.image_shape[:2], 1)
        self.epoch, self.loss, self.acc = [], [], []
        
        self.model = self.combined()
        self.model.compile(self.optimizer, 'mse', metrics=['accuracy'])
    
    def resnet_block(self, input_image):
        x = layers.Conv2D(self.res_filters, (3,3), padding='same')(input_image)
        x = InstanceNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(self.res_filters, (3,3), padding='same')(x)
        x = InstanceNormalization(axis=-1)(x)
        
        x = layers.Concatenate()([x, input_image])
        return x
    
    def combined(self):
        
        def conv2d(input_layer, filters, kernel, strides):
            x = layers.Conv2D(filters, kernel, strides=strides, padding='same')(input_layer)
            x = InstanceNormalization(axis=-1)(x)
            x = layers.Activation('relu')(x)
            return x
        
        def conv2dt(input_layer, filters, kernel, strides):
            x = layers.Conv2DTranspose(filters, kernel, strides=strides, padding='same')(input_layer)
            x = InstanceNormalization(axis=-1)(x)
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
        x = InstanceNormalization(axis=-1)(x)
        output_image = layers.Activation('tanh')(x)
        
        return Model(input_image, output_image)


class CNN3(Utils):
    def __init__(
        self,
        run_name,
        image_shape = (256,256,3),
        batch_size = 1,
        res_filters = 512,
        conv_filters = 64,
        nb_resnet = 20,
        optimizer = Adam(),
        data_generator = DataGenerator((256,256), 1),
        epoch_save_interval = 10
    ):
        self.run_name = run_name
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.res_filters = res_filters
        self.conv_filters = conv_filters
        self.nb_resnet = nb_resnet
        self.optimizer = optimizer
        self.data_generator = data_generator
        self.epoch_save_interval = epoch_save_interval
        
        self.init_weights = RandomNormal()
        
        self.sample_generator = DataGenerator(self.image_shape[:2], 1)
        self.epoch, self.loss, self.acc = [], [], []
        
        self.model = self.combined()
        self.model.compile(self.optimizer, 'mse', metrics=['accuracy'])
    
    def resnet_block(self, input_image):
        x = layers.Conv2D(self.res_filters, (3,3), padding='same', kernel_initializer=self.init_weights)(input_image)
        x = InstanceNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(self.res_filters, (3,3), padding='same', kernel_initializer=self.init_weights)(x)
        x = InstanceNormalization(axis=-1)(x)
        
        x = layers.Concatenate()([x, input_image])
        return x
    
    def combined(self):
        
        def conv2d(input_layer, filters, kernel, strides):
            x = layers.Conv2D(filters, kernel, strides=strides, padding='same', kernel_initializer=self.init_weights)(input_layer)
            x = InstanceNormalization(axis=-1)(x)
            x = layers.Activation('relu')(x)
            return x
        
        def conv2dt(input_layer, filters, kernel, strides):
            x = layers.Conv2DTranspose(filters, kernel, strides=strides, padding='same', kernel_initializer=self.init_weights)(input_layer)
            x = InstanceNormalization(axis=-1)(x)
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
        
        x = layers.Conv2D(3, (7,7), padding='same', kernel_initializer=self.init_weights)(x)
        x = InstanceNormalization(axis=-1)(x)
        output_image = layers.Activation('tanh')(x)
        
        return Model(input_image, output_image)


class CNN4(Utils):
    def __init__(
        self,
        run_name,
        image_shape = (256,256,3),
        batch_size = 2,
        res_filters = 256,
        conv_filters = 64,
        nb_resnet = 18,
        optimizer = Adam(),
        data_generator = DataGenerator((256,256), 2),
        epoch_save_interval = 10
    ):
        self.run_name = run_name
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.res_filters = res_filters
        self.conv_filters = conv_filters
        self.nb_resnet = nb_resnet
        self.optimizer = optimizer
        self.data_generator = data_generator
        self.epoch_save_interval = epoch_save_interval
        
        self.init_weights = RandomNormal()
        
        self.sample_generator = DataGenerator(self.image_shape[:2], 1)
        self.epoch, self.loss, self.acc = [], [], []
        
        self.model = self.combined()
        self.model.compile(self.optimizer, 'mae', metrics=['accuracy'])
    
    def resnet_block(self, input_image):
        x = layers.Conv2D(self.res_filters, (3,3), padding='same', kernel_initializer=self.init_weights)(input_image)
        x = InstanceNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(self.res_filters, (3,3), padding='same', kernel_initializer=self.init_weights)(x)
        x = InstanceNormalization(axis=-1)(x)
        
        x = layers.Concatenate()([x, input_image])
        return x
    
    def combined(self):
        
        def conv2d(input_layer, filters, kernel, strides):
            x = layers.Conv2D(filters, kernel, strides=strides, padding='same', kernel_initializer=self.init_weights)(input_layer)
            x = InstanceNormalization(axis=-1)(x)
            x = layers.Activation('relu')(x)
            return x
        
        def conv2dt(input_layer, filters, kernel, strides):
            x = layers.Conv2DTranspose(filters, kernel, strides=strides, padding='same', kernel_initializer=self.init_weights)(input_layer)
            x = InstanceNormalization(axis=-1)(x)
            x = layers.Activation('relu')(x)
            return x
        
        input_image = layers.Input(shape=self.image_shape)

        x = conv2d(input_image, self.conv_filters*1, (7,7), (1,1))
        x = conv2d(x, self.conv_filters*2, (3,3), (2,2))
        x = conv2d(x, self.conv_filters*4, (3,3), (2,2))
        x = conv2d(x, self.conv_filters*8, (3,3), (2,2))
        
        for _ in range(self.nb_resnet):
            x = self.resnet_block(x)
        
        x = conv2dt(x, self.conv_filters*4, (3,3), (2,2))
        x = conv2dt(x, self.conv_filters*2, (3,3), (2,2))
        x = conv2dt(x, self.conv_filters*1, (3,3), (2,2))
        
        x = layers.Conv2D(3, (7,7), padding='same', kernel_initializer=self.init_weights)(x)
        x = InstanceNormalization(axis=-1)(x)
        output_image = layers.Activation('tanh')(x)
        
        return Model(input_image, output_image)
