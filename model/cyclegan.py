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

from src.data_generator_train_v5 import DataGenerator_trainv5

class Utils():
    
    def export_sample(self, epoch, name):
        plt.rcParams["figure.autolayout"] = True
        
        img_A, img_B = self.sample_loader.__getitem__(0)
        fake_B = self.g_AB.predict(img_A, verbose=0)
        fake_A = self.g_BA.predict(img_B, verbose=0)
        
        plt.figure(figsize=(20,40))
        for i, (iA, iB, fA, fB) in enumerate(zip(img_A, img_B, fake_A, fake_B)):
            plt.subplot(6,4,(i*4)+1)
            plt.title("REAL NIGHT")
            plt.imshow(iA.astype(np.float64))
            plt.subplot(6,4,(i*4)+2)
            plt.title("REAL DAY")
            plt.imshow(iB.astype(np.float64))
            plt.subplot(6,4,(i*4)+3)
            plt.title("FAKE NIGHT")
            plt.imshow(fA.astype(np.float64))
            plt.subplot(6,4,(i*4)+4)
            plt.title("FAKE DAY")
            plt.imshow(fB.astype(np.float64))
        plt.savefig(f"../resources/sample/cyclegan/{self.__class__.__name__}_{name}_e{epoch}_r({self.img_rows},{self.img_cols})_b{self.batch_size}.jpg")
        plt.close()

    def export_model(self, epoch, name):
        if (epoch+1) % 1 == 0:
            self.combined.save(f"../model/saved_model/CycleGAN/{self.__class__.__name__}_{name}_e{epoch}_r({self.img_rows},{self.img_cols})_b{self.batch_size}.keras")

    def export_reports(self, name):
        pd.DataFrame({
                    "epoch": self.epoch, 
                    "dis_loss": self.dis_loss, 
                    "dis_acc": self.dis_acc, 
                    "gen_loss": self.gen_loss, 
                    "adv": self.adv, 
                    "recon": self.recon, 
                    "id": self.id, 
                    "time": self.time
        }).to_csv(f"../resources/reports/cyclegan/{self.__class__.__name__}_{name}_r({self.img_rows},{self.img_cols})_b{self.batch_size}.csv")
        
    def train(self, epochs, name):

        self.export_sample(0, name)
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader):
                
                # ----------------------
                #  Train Discriminators
                # ----------------------

                if (batch_i+1) % self.nb_dis_train == 0:
                    # Translate images to opposite domain
                    fake_B = self.g_AB.predict(imgs_A, verbose=0)
                    fake_A = self.g_BA.predict(imgs_B, verbose=0)

                    # Train the discriminators (original images = real / translated = Fake)
                    dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                    dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                    dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                    dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                    # Total disciminator loss
                    d_loss = 0.5 * np.add(dA_loss, dB_loss)
                else:
                    d_loss = [np.nan, np.nan]

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                self.epoch.append(epoch + ( (batch_i+1)/self.data_loader.nb_batch ) )
                self.dis_loss.append(d_loss[0])
                self.dis_acc.append(d_loss[1])
                self.gen_loss.append(g_loss[0])
                self.adv.append(np.mean(g_loss[1:3]))
                self.recon.append(np.mean(g_loss[3:5]))
                self.id.append(np.mean(g_loss[5:6]))
                self.time.append(elapsed_time)
                
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch+1, epochs,
                                                                            batch_i+1, self.data_loader.nb_batch,
                                                                            d_loss[0] if not np.isnan(d_loss[0]) else 0, 100*d_loss[1] if not np.isnan(d_loss[1]) else 0,
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

            # If at save interval => save generated image samples
            if (epoch+1) % self.log_interval == 0:
                self.export_sample(epoch+1, name)
                self.export_model(epoch+1, name)

class CycleGAN(Utils):
    def __init__(self):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.batch_size = 2
        self.log_interval = 2

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.epoch, self.dis_loss, self.dis_acc, self.gen_loss, self.adv, self.recon, self.id, self.time = [], [], [], [], [], [], [], []

        # Configure data loader
        self.data_loader = DataGenerator_trainv3(self.img_shape[:2], self.batch_size)

        self.sample_loader = DataGenerator_trainv3(self.img_shape[:2], 3)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.001, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
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
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)


class CycleGANResNet(Utils):
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.batch_size = 2
        self.log_interval = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.epoch, self.dis_loss, self.dis_acc, self.gen_loss, self.adv, self.recon, self.id, self.time = [], [], [], [], [], [], [], []

        # Configure data loader
        self.data_loader = DataGenerator_trainv3(self.img_shape[:2], self.batch_size)

        self.sample_loader = DataGenerator_trainv3(self.img_shape[:2], 3)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64
        self.rf = 128
        self.nb_resnet = 3
        self.nb_block_resnet = 4

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """Res-Net Generator"""
        
        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            return u

        def resnet_block(layer_input, filters, f_size=3):
            x = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            
            x = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            return Concatenate()([x, layer_input])

        # Image input
        input_img = Input(shape=self.img_shape)

        # Downsampling
        x = conv2d(input_img, self.gf)
        x = conv2d(x, self.gf*2)
        x = conv2d(x, self.gf*4)

        for i in range(self.nb_block_resnet):
            for _ in range(self.nb_resnet):
                x = resnet_block(x, self.rf*(i+1))

        # Upsampling
        x = deconv2d(x, self.gf*2)
        x = deconv2d(x, self.gf)

        x = UpSampling2D(size=2)(x)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

        return Model(input_img, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)


class CycleGANResNetv2(Utils):
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.batch_size = 2
        self.log_interval = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.epoch, self.dis_loss, self.dis_acc, self.gen_loss, self.adv, self.recon, self.id, self.time = [], [], [], [], [], [], [], []

        # Configure data loader
        self.data_loader = DataGenerator_trainv3(self.img_shape[:2], self.batch_size)

        self.sample_loader = DataGenerator_trainv3(self.img_shape[:2], 3)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 16
        self.df = 64
        self.rf = 128
        self.nb_resnet = 30
        self.nb_block_resnet = 1

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """Res-Net Generator"""
        
        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            return u

        def resnet_block(layer_input, filters, f_size=3):
            x = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            
            x = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            return Concatenate()([x, layer_input])

        # Image input
        input_img = Input(shape=self.img_shape)

        # Downsampling
        x = conv2d(input_img, self.gf, f_size=7)
        x = conv2d(x, self.gf*2)
        x = conv2d(x, self.gf*4)
        x = conv2d(x, self.gf*8)

        for i in range(self.nb_block_resnet):
            for _ in range(self.nb_resnet):
                x = resnet_block(x, self.rf*(i+1))

        x = deconv2d(x, self.gf*4)
        x = deconv2d(x, self.gf*2)
        x = deconv2d(x, self.gf)

        # Upsampling
        x = UpSampling2D(size=2)(x)
        output_img = Conv2D(self.channels, kernel_size=7, strides=1, padding='same', activation='tanh')(x)

        return Model(input_img, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)
    
class CycleGANResNetUNet(Utils):
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.batch_size = 5
        self.log_interval = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.epoch, self.dis_loss, self.dis_acc, self.gen_loss, self.adv, self.recon, self.id, self.time = [], [], [], [], [], [], [], []

        # Configure data loader
        self.data_loader = DataGenerator_trainv3(self.img_shape[:2], self.batch_size)

        self.sample_loader = DataGenerator_trainv3(self.img_shape[:2], 3)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64
        self.rf = 256
        self.nb_resnet = 9
        self.nb_block_resnet = 1

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """Res-Net Generator"""
        
        def conv2d(layer_input, filters, f_size=3):
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

        def resnet_block(layer_input, filters, f_size=3):
            x = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            
            x = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            return Concatenate()([x, layer_input])

        # Image input
        input_img = Input(shape=self.img_shape)

        # Downsampling
        d0 = conv2d(input_img, self.gf)
        d1 = conv2d(d0, self.gf*2)
        d2 = conv2d(d1, self.gf*4)
        d3 = conv2d(d2, self.gf*8)

        for i in range(self.nb_block_resnet):
            for _ in range(self.nb_resnet):
                d3 = resnet_block(d3, self.rf*(i+1))

        u2 = deconv2d(d3, d2, self.gf*4)
        u1 = deconv2d(u2, d1, self.gf*2)
        u0 = deconv2d(u1, d0, self.gf)

        # Upsampling
        x = UpSampling2D(size=2)(u0)
        output_img = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        return Model(input_img, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)
    
class CycleGANUnet(Utils):
    def __init__(self):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.batch_size = 2
        self.log_interval = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.epoch, self.dis_loss, self.dis_acc, self.gen_loss, self.adv, self.recon, self.id, self.time = [], [], [], [], [], [], [], []

        # Configure data loader
        self.data_loader = DataGenerator_trainv3(self.img_shape[:2], self.batch_size)

        self.sample_loader = DataGenerator_trainv3(self.img_shape[:2], 3)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """Res-Net Generator"""
        
        def conv2d(layer_input, filters, f_size=3):
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
        input_img = Input(shape=self.img_shape)

        # Downsampling
        d0 = conv2d(input_img, self.gf)
        d1 = conv2d(d0, self.gf*2)
        d2 = conv2d(d1, self.gf*4)
        d3 = conv2d(d2, self.gf*8)
        d4 = conv2d(d3, self.gf*16)

        u3 = deconv2d(d4, d3, self.gf*8)
        u2 = deconv2d(u3, d2, self.gf*4)
        u1 = deconv2d(u2, d1, self.gf*2)
        u0 = deconv2d(u1, d0, self.gf)

        # Upsampling
        x = UpSampling2D(size=2)(u0)
        output_img = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        return Model(input_img, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)
    

class CycleGANUnetv2(Utils):
    def __init__(self):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.batch_size = 1
        self.log_interval = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.epoch, self.dis_loss, self.dis_acc, self.gen_loss, self.adv, self.recon, self.id, self.time = [], [], [], [], [], [], [], []

        # Configure data loader
        self.data_loader = DataGenerator_trainv5(self.img_shape[:2], self.batch_size)

        self.sample_loader = DataGenerator_trainv5(self.img_shape[:2], 6)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64
        self.rf = 512
        self.nb_resnet = 3

        self.nb_dis_train = 2

        # Loss weights
        self.lambda_cycle = 2                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss
        self.genloss_weight = 2
        self.disloss_weights = 0.5

        optimizer = Adam(0.0003, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'],
            loss_weights=[self.disloss_weights])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'],
            loss_weights=[self.disloss_weights])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  self.genloss_weight, self.genloss_weight,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """Res-Net Generator"""
        
        def conv2d(layer_input, filters, f_size=3):
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

        def resnet_block(layer_input, filters, f_size=3):
            x = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            
            x = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
            x = InstanceNormalization(axis=-1)(x)
            return Concatenate()([x, layer_input])
        
        # Image input
        input_img = Input(shape=self.img_shape)

        # Downsampling
        d0 = conv2d(input_img, self.gf)
        d1 = conv2d(d0, self.gf*2)
        d2 = conv2d(d1, self.gf*4)
        d3 = conv2d(d2, self.gf*8)

        for _ in range(self.nb_resnet):
            d3 = resnet_block(d3, self.rf)

        # Upsampling
        u2 = deconv2d(d3, d2, self.gf*4)
        u1 = deconv2d(u2, d1, self.gf*2)
        u0 = deconv2d(u1, d0, self.gf)
        
        x = UpSampling2D(size=2)(u0)
        output_img = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        return Model(input_img, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)