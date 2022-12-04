import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_graph(img_day, img_night):
    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    plt.title("DAY")
    plt.imshow(img_day)
    plt.subplot(1,2,2)
    plt.title("NIGHT")
    plt.imshow(img_night)
    plt.show(block=False)

def load_image_day_night(name: str):
    return (
        mpimg.imread(f"../data/timelaps/day/{name}.jpeg"),
        mpimg.imread(f"../data/timelaps/night/{name}.jpeg")
    )
    
def get_user_interaction():
    user_input = input()
    
    if user_input == "1":
        print("1")
    if user_input == "2":
        print("2")
        
    plt.close()

def main():
    for name in range(0,3):
        create_graph(*load_image_day_night(name))
        get_user_interaction()

if __name__ == "__main__":
    plt.rcParams["figure.autolayout"] = True
    main()