from PIL import Image
import numpy as np
import cv2
from os import walk

def load_process_image(path):
    image = Image.open(path)
    np_image = np.asarray(image)/255
    return cv2.resize(np_image, (256,256)).astype(np.float32)

def get_file_names(dir):
    for _,_,file_names in walk(dir):
        return file_names

def main():
    day_name = get_file_names(OLD_DIR + "day")
    night_name = get_file_names(OLD_DIR + "night")

    for i, name in enumerate(day_name):
        print(f"DAY : {i}")
        clean_img = load_process_image(OLD_DIR + "day/" + name)
        np.save(NEW_DIR + "day/" + name, clean_img)
    for i, name in enumerate(night_name):
        print(f"NIGHT : {i}")
        clean_img = load_process_image(OLD_DIR + "night/" + name)
        np.save(NEW_DIR + "night/" + name, clean_img)

if __name__ == "__main__":
    OLD_DIR = "../data/train_v3/"
    NEW_DIR = "../data/256_preprocess_train_v3/"
    main()