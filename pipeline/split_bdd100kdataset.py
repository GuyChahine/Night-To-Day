import numpy as np
from os import walk
from shutil import copy


def get_file_name(dir):
    for _,_,file_name in walk(dir):
        return np.array(file_name)

def copy_files(old_dir, new_dir, file_names):
    for name in file_names:
        copy(old_dir + name, new_dir + name)

def main():
    name_day = get_file_name(OLD_DIR + "trainA")
    name_night = get_file_name(OLD_DIR + "trainB")
    
    copy_files(OLD_DIR + "trainA/", NEW_DIR + "day/", np.random.choice(name_day, SIZE))
    copy_files(OLD_DIR + "trainB/", NEW_DIR + "night/", np.random.choice(name_night, SIZE))

if __name__ == "__main__":
    SIZE = 500
    OLD_DIR = "../data/bdd100kdataset/bdd100k/bdd100k/images/100k/train/"
    NEW_DIR = "../data/clean_bdd100kdataset/"
    main()