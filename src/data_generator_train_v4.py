import numpy as np
from keras.utils import Sequence
from os import walk
import cv2
from PIL import Image

class DataGenerator_trainv4(Sequence):
    'Generates data for Keras'
    def __init__(self, image_size=(512,512), batch_size=4, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = image_size
        self.dir_day = "../data/1024_preprocess_train_v4/day/"
        self.dir_night = "../data/1024_preprocess_train_v4/night/"
        self.name = self.get_file_name(self.dir_night)
        self.nb_batch = len(self.name)//batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.name) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        temp_name = self.name[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        x, y = self.__data_generation(temp_name)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.name)
            
    def get_file_name(self, dir):
        for _,_,file_name in walk(dir):
            return np.array(file_name)

    def __data_generation(self, name):
        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        night_batch = np.array([np.load(self.dir_night + name) for name in name])
        day_batch = np.array([np.load(self.dir_day + name) for name in name])

        return night_batch, day_batch