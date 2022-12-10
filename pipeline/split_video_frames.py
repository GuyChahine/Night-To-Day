import cv2
from argparse import ArgumentParser

def mp4_to_images(name: str):
    capture = cv2.VideoCapture(f"../data/youtube_timelaps/video/{name}.mp4")
    nb_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    frame_count = 0
    while True:
        print(f"{frame_count}/{nb_frame}", end="\r")
        con, frames = capture.read()
        if con:
            if (frame_count % 100) == 0:
                cv2.imwrite(f"../data/youtube_timelaps/{name}_frames/{name}_{frame_count}.jpeg", frames)
            frame_count += 1
        else:
            break
        
def main(name):
    mp4_to_images(name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-name', help='Give the name of the video', required=True)
    
    main(**vars(parser.parse_args()))