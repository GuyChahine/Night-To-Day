import numpy as np

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import sys
sys.path.insert(0, "\\".join(__file__.split("\\")[:__file__.split("\\").index("Night-To-Day")+1]))

from src.data_generator import DataGenerator

class CycleGAN():
    def __init__(
        self,
    ):
        self.image_shape = (128,128,3)
        self.batch_size = 5
        
        self.data_generator = DataGenerator(image_size=self.image_shape[:2], batch_size=self.batch_size, shuffle=True)
        
        self.g_filter = 64
        self.d_filter = 64
        self.r_filter = 256
        self.n_resnet = 9
        
        self.patch = int(self.image_shape[1] / 2**5)
        
        optimizer = Adam(0.0002, 0.5)
        
        self.dA = self.build_discriminator()
        self.dB = self.build_discriminator()
        
        self.dA.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.dB.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        
        self.gAB = self.build_generator()
        self.gBA = self.build_generator()
        
        self.dA.trainable = False
        self.dB.trainable = False
        
        image_A = Input(self.image_shape)
        image_B = Input(self.image_shape)
        
        fake_B = self.gAB(image_A)
        fake_A = self.gBA(image_B)
        reconstr_A = self.gBA(fake_B)
        reconstr_B = self.gAB(fake_A)
        imageA_id = self.gBA(image_A)
        imageB_id = self.gAB(image_B)
    
        valid_A = self.dA(fake_A)
        valid_B = self.dB(fake_B)
        
        self.combined = Model([image_A, image_B],
                              [valid_A, valid_B,
                               reconstr_A, reconstr_B,
                               imageA_id, imageB_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1,1,
                                            10,10,
                                            1,1],
                              optimizer=optimizer)
    
    def resnet_block(self, input_layer):
        
        x = Conv2D(self.r_filter, (3,3), padding='same')(input_layer)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        
        x = Conv2D(self.r_filter, (3,3), padding='same')(x)
        x = InstanceNormalization(axis=-1)(x)
        
        x = Concatenate()([x, input_layer])
        return x
    
    def build_generator(self):
        
        def conv2d(layer_input, filters, kernel, strides):
            x = Conv2D(filters, kernel, strides=strides, padding='same')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            return x
        
        def conv2dt(layer_input, filters, kernel, strides):
            x = Conv2DTranspose(filters, kernel, strides=strides, padding='same')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            x = Activation('relu')(x)
            return x
        
        input_image = Input(shape=self.image_shape)
        
        x = conv2d(input_image, self.g_filter*1, (7,7), (1,1))
        x = conv2d(x, self.g_filter*2, (3,3), (2,2))
        x = conv2d(x, self.g_filter*4, (3,3), (2,2))
        
        for _ in range(self.n_resnet):
            x = self.resnet_block(x)
        
        x = conv2dt(x, self.g_filter*2, (3,3), (2,2))
        x = conv2dt(x, self.g_filter*1, (3,3), (2,2))
        
        x = Conv2D(3, (7,7), padding='same')(x)
        x = InstanceNormalization(axis=-1)(x)
        output_image = Activation('tanh')(x)
        
        model = Model(input_image, output_image)
        return model
    
    def build_discriminator(self):
        
        def conv2d(layer_input, filters, kernel, strides):
            x = Conv2D(filters, kernel, strides=strides, padding='same')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            x = LeakyReLU(alpha=0.2)(x)
            return x
        
        input_image = Input(shape=self.image_shape)
        
        x = Conv2D(self.d_filter, (4,4), strides=(2,2), padding='same')(input_image)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = conv2d(x, self.d_filter*2, (4,4), (2,2))
        x = conv2d(x, self.d_filter*4, (4,4), (2,2))
        x = conv2d(x, self.d_filter*8, (4,4), (2,2))
        x = conv2d(x, self.d_filter*8, (4,4), (2,2))
        
        validity = Conv2D(1, (4,4), padding='same')(x)
        
        model = Model(input_image, validity)
        return model
    
    def train(self, epochs):
        
        valid = np.ones((self.batch_size,) + (self.patch, self.patch, 1))
        fake = np.zeros((self.batch_size,) + (self.patch, self.patch, 1))
        
        for epoch in range(epochs):
            for batch_i, (image_A, image_B) in enumerate(self.data_generator):
                fake_B = self.gAB(image_A)
                fake_A = self.gBA(image_B)
                
                dA_loss_real = self.dA.train_on_batch(image_A, valid)
                dA_loss_fake = self.dA.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                
                dB_loss_real = self.dB.train_on_batch(image_B, valid)
                dB_loss_fake = self.dB.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                
                d_loss = 0.5 * np.add(dA_loss, dB_loss)
                
                g_loss = self.combined.train_on_batch([image_A, image_B],
                                                      [valid, valid,
                                                       image_A, image_B,
                                                       image_A, image_B])
                
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f]  " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_generator.nb_batch,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6])))
                
cyclegan = CycleGAN()
cyclegan.train(2)