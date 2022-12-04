import requests
import cv2
import re

# From mp4 link to folder
def mp4url_downloader(name: str, url: str):
    with open(f"data/timelaps/video/{name}.mp4", "wb") as f:
        f.write(requests.get(url).content)

# From mp4 to first few frame image and last few frame image  
def mp4_to_images(name: str):
    capture = cv2.VideoCapture(f"data/timelaps/video/{name}.mp4")
    nb_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    frame_count = 0
    while True:
        con, frames = capture.read()
        if con:
            if frame_count == 20:
                cv2.imwrite(f"data/timelaps/day/{name}.jpeg", frames)
            elif frame_count == (nb_frame - 50):
                cv2.imwrite(f"data/timelaps/night/{name}.jpeg", frames)
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

if __name__ == "__main__":
    
    BASE_URL = "https://www.videvo.net/search/day%20to%20night/"
    
    mp4_links = []
    for i, link in enumerate(videvo_find_links(BASE_URL)):
        print(i, end="\r")
        mp4_links.append(videvo_find_mp4(link))
    
    for i, link in enumerate(mp4_links):
        print(i, end="\r")
        mp4url_downloader(str(i), link)
        mp4_to_images(str(i))