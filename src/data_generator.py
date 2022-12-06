import numpy as np
from keras.utils import Sequence
from os import walk
import cv2
from PIL import Image

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, image_size=(256,256), batch_size=32, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.image_size = image_size
        for _,_,file_names in walk("../data/train_v1/day/"):
            self.list_IDs = file_names
        self.nb_batch = len(self.list_IDs)//batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        def process_image(image):
            np_image = np.asarray(image)/255
            return cv2.resize(np_image, self.image_size)
        
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        x = np.array([process_image(Image.open(f"../data/train_v1/night/{name}")) for name in  list_IDs_temp])
        y = np.array([process_image(Image.open(f"../data/train_v1/day/{name}")) for name in  list_IDs_temp])

        return x, y