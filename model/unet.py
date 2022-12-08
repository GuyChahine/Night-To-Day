from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, LeakyReLU, Dropout, Concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("Night-To-Day")+1]))

from src.data_generator_train_v4 import DataGenerator_trainv4


class Utils():
    
    def export_sample(self, epoch, name):
        plt.rcParams["figure.autolayout"] = True
        
        img_A, img_B = self.sample_loader.__getitem__(0)
        fake_B = self.gen.predict(img_A, verbose=0)
        
        plt.figure(figsize=(20,20))
        for i, (iA, iB, fB) in enumerate(zip(img_A, img_B, fake_B)):
            plt.subplot(3,3,(i*3)+1)
            plt.title("REAL NIGHT")
            plt.imshow(iA.astype(np.float64))
            plt.subplot(3,3,(i*3)+2)
            plt.title("REAL DAY")
            plt.imshow(iB.astype(np.float64))
            plt.subplot(3,3,(i*3)+3)
            plt.title("FAKE DAY")
            plt.imshow(fB.astype(np.float64))
        plt.savefig(f"../resources/sample/unet/{self.__class__.__name__}_{name}_e{epoch}_r({self.img_rows},{self.img_cols})_b{self.batch_size}.jpg")
        plt.close()

    def export_model(self, epoch, name):
        if epoch % 10 == 0:
            self.gen.save(f"../model/saved_model/unet/{self.__class__.__name__}_{name}_e{epoch}_r({self.img_rows},{self.img_cols})_b{self.batch_size}.keras")

    def export_report(self, name):
        pd.DataFrame({
            "epoch": self.epoch,
            "loss": self.loss,
            "acc": self.acc
        }).to_csv(f"../resources/reports/unet/{name}.csv")
    
    def train(self, epochs, name):

        self.export_sample(0, name)
        for epoch in range(epochs):
            for batch_i, (image_night, image_day) in enumerate(self.data_loader):
                model_loss = self.gen.train_on_batch(image_night, image_day)
                
                print("[Epoch {}/{}] [Batch {}/{}] [Loss {:.5f}] [Accuracy {:.5f}]".format(
                    epoch+1, epochs,
                    batch_i+1, self.data_loader.nb_batch,
                    model_loss[0], model_loss[1]
                ))
                
                self.epoch.append( epoch + ( (batch_i+1)/self.data_loader.nb_batch ) )
                self.loss.append(model_loss[0])
                self.acc.append(model_loss[1])
                
            if ( (epoch+1) % self.log_interval) == 0:
                self.export_sample(epoch+1, name)
                self.export_model(epoch+1, name)

class UNET(Utils):
    def __init__(self):
        # Input shape
        self.img_rows = 1024
        self.img_cols = 1024
        self.channels = 3
        self.batch_size = 1
        self.log_interval = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.epoch, self.loss, self.acc = [], [], []

        # Configure data loader
        self.data_loader = DataGenerator_trainv4(self.img_shape[:2], self.batch_size)

        self.sample_loader = DataGenerator_trainv4(self.img_shape[:2], 3)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64

        optimizer = Adam(0.0002, 0.5)

        # Build the generators
        self.gen = self.build_generator()

        self.gen.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u3 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u3, d2, self.gf*2)
        u1 = deconv2d(u2, d1, self.gf)

        x = UpSampling2D(size=2)(u1)
        output_img = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        return Model(d0, output_img)