import numpy as np
from keras.utils import Sequence
from os import walk
import cv2
from PIL import Image

class DataGenerator_trainv2(Sequence):
    'Generates data for Keras'
    def __init__(self, image_size=(256,256), batch_size=4, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = image_size
        self.dir_day = "../data/train_v2/day/"
        self.dir_night = "../data/train_v2/night/"
        self.name_day = self.get_file_name(self.dir_day)
        self.name_night = self.get_file_name(self.dir_night)
        self.nb_batch = len(self.name_day)//batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.name_day) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        temp_name_day = self.name_day[index*self.batch_size:(index+1)*self.batch_size]
        temp_name_night = self.name_night[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        x, y = self.__data_generation(temp_name_day, temp_name_night)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.name_day)
            np.random.shuffle(self.name_night)
            
    def get_file_name(self, dir):
        for _,_,file_name in walk(dir):
            return np.array(file_name)

    def __data_generation(self, name_day, name_night):
        
        def process_image(image):
            np_image = np.asarray(image)/255
            return cv2.resize(np_image, self.image_size)
        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        night_batch = np.array([process_image(Image.open(self.dir_night + name)) for name in name_night])
        day_batch = np.array([process_image(Image.open(self.dir_day + name)) for name in name_day])

        return night_batch, day_batch