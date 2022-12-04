import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import move
from os import remove, walk
import cv2

def delete_image(name: str):
    remove(f"../data/timelaps/day/{name}.jpeg")
    remove(f"../data/timelaps/night/{name}.jpeg")

# The have the day image in day folder and night image in night folder    
def move_image_normal(name: str):
    move(f"../data/timelaps/day/{name}.jpeg", f"../data/clean_timelaps/day/{name}.jpeg")
    move(f"../data/timelaps/night/{name}.jpeg", f"../data/clean_timelaps/night/{name}.jpeg")
    
# The have the day image in night folder and night image in day folder so we put it in the right place  
def move_image_inverse(name: str):
    move(f"../data/timelaps/day/{name}.jpeg", f"../data/clean_timelaps/night/{name}.jpeg")
    move(f"../data/timelaps/night/{name}.jpeg", f"../data/clean_timelaps/day/{name}.jpeg")

# Create graph to visualise pictures
def create_graph(img_day, img_night):
    plt.figure(figsize=(20,8))
    plt.subplot(2,2,1)
    plt.title("DAY")
    plt.imshow(img_day)
    plt.subplot(2,2,2)
    plt.title("NIGHT")
    plt.imshow(img_night)
    plt.subplot(2,2,3)
    plt.title("DIFF")
    plt.imshow(cv2.subtract(img_day, img_night))
    plt.show(block=False)

def load_image_day_night(name: str):
    return (
        mpimg.imread(f"../data/timelaps/day/{name}.jpeg"),
        mpimg.imread(f"../data/timelaps/night/{name}.jpeg")
    )

# Function to get what user wan't to do
def get_user_interaction(name: str):
    user_input = input()
    
    # User can press on 
    # 0 if the data is ok
    # 1 if the data is inverted
    # 2 if the data is not good
    if user_input == "0":
        move_image_normal(name)
    elif user_input == "1":
        move_image_inverse(name)
    elif user_input == "2":
        delete_image(name)
        
    plt.close()

def get_existing_file_names():
    for (_dir_path, _dir_names, file_names) in walk("../data/timelaps/day/"):
        return [name.replace(".jpeg", "") for name in file_names]

def main():
    
    for name in get_existing_file_names()[0:20]:
        create_graph(*load_image_day_night(name))
        get_user_interaction(name)

if __name__ == "__main__":
    plt.rcParams["figure.autolayout"] = True
    main()