import requests
import cv2
import re
from os import walk
import pickle
from os.path import exists
from argparse import ArgumentParser

# From mp4 link to folder
def mp4url_downloader(name: str, url: str):
    with open(f"../data/timelaps/video/{name}.mp4", "wb") as f:
        f.write(requests.get(url).content)

# From mp4 to first few frame image and last few frame image  
def mp4_to_images(name: str):
    capture = cv2.VideoCapture(f"../data/timelaps/video/{name}.mp4")
    nb_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    frame_count = 0
    while True:
        con, frames = capture.read()
        if con:
            if frame_count == 20:
                cv2.imwrite(f"../data/timelaps/day/{name}.jpeg", frames)
            elif frame_count == (nb_frame - 50):
                cv2.imwrite(f"../data/timelaps/night/{name}.jpeg", frames)
            frame_count += 1
        else:
            break
        
# Scrap videvo page for page of timelaps links
def videvo_find_links(query_link: str):
    r = requests.get(query_link)
    return [f"https://www.videvo.net/{link}" for link in re.findall("""<a href="(/video/.*?/)">""", r.text)]

# Scrap videvo page for timelaps mp4 link
def videvo_find_mp4(page_link: str):
    r = requests.get(page_link)
    mp4_link = re.findall("""src="(https://.*?.mp4)">""", r.text)
    assert len(mp4_link) == 1, "Found multiple link in timelaps page"
    return mp4_link[0]

# Get the last image number in the day folder
def get_nb_last_existing_images():
    for (_dir_path, _dir_names, file_names) in walk("../data/timelaps/day/"):
        images_number = [int(name.replace(".jpeg", "")) for name in file_names]
        max_unclean = max(images_number)+1 if len(images_number) > 0 else 0
    for (_dir_path, _dir_names, file_names) in walk("../data/clean_timelaps/day/"):
        images_number = [int(name.replace(".jpeg", "")) for name in file_names]
        max_clean = max(images_number)+1 if len(images_number) > 0 else 0
    return max_unclean if max_unclean >= max_clean else max_clean

def save_links(links: list):
    with open("../data/timelaps/links_tracker.pickle", "wb") as f:
        pickle.dump(links, f)
        
def load_links():
    if exists("../data/timelaps/links_tracker.pickle"):
        with open("../data/timelaps/links_tracker.pickle", "rb") as f:
            return pickle.load(f)
    else:
        return []

# Check if the links has been already used in the dataset
def get_unique_links(links_used: list, new_links: list):
    return [nl for nl in new_links if nl not in links_used]

def main(base_url: list):

    # Get the link of the video page
    page_links = [page_link for url in base_url for page_link in videvo_find_links(url)]
    
    # Look for mp4 link of the video
    new_links = []
    for i, link in enumerate(page_links):
        print(f"SEARCH FOR MP4 LINKS : {i}/{len(page_links)}", end="\r")
        new_links.append(videvo_find_mp4(link))
    print()
    
    # Check if link is already downloaded
    links_used = load_links()
    mp4_links = get_unique_links(links_used, new_links)
    
    # Download and save the image
    last_image_number = get_nb_last_existing_images()
    for i, link in enumerate(mp4_links):
        print(f"DOWNLOAD AND CUTING IMAGES : {i}/{len(mp4_links)}", end="\r")
        mp4url_downloader(str(i + last_image_number), link)
        mp4_to_images(str(i + last_image_number))
    print()
        
    save_links(links_used + mp4_links)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-base_url', nargs='+', help='Give the links of the query', required=True)
    
    main(**vars(parser.parse_args()))