import pafy 
import googleapiclient.discovery
import pandas as pd
from moviepy.editor import *
import os


def read_excel_database(filename):
    df = pd.read_csv(filename)
    id = df["index"]
    artist = df['artist']
    title = df['title']
    return id, artist, title


def scrape_link(phrase):
    api_service_name = "youtube"
    api_version = "v3"
    api_key = "" #insert your api key
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=api_key)
    search_phrase = phrase

    search_response = youtube.search().list(
            q=search_phrase,
            part="id,snippet",
            maxResults=1
        ).execute()

    video_id = search_response["items"][0]["id"]["videoId"]
    link = f"https://www.youtube.com/watch?v={video_id}"

    print(f"{search_phrase}'; {link}")
    return link


def download_video(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    best.download(quiet=False)
    return video.title+".mp4"
    
    
def save_index_name_and_link(filename, index, name, link, opt=""):
    with open(filename, 'a') as f:
        f.write(f'{index}_{name}_{link}_{opt}')
        f.write("\n")
        
        
def convert_to_mp3_and_change_name(filename, index):
    video = VideoFileClip(filename)
    video.audio.write_audiofile(f"{index}.mp3")
    
    
def remove_all_mp4_files(directory):
    for file in os.listdir(directory):
        if(file.endswith(".mp4")):
            os.remove(file)
    
    
if __name__=="__main__":
    ids, artists, titles = read_excel_database("./MoodyLyrics4Q.csv")
    for id, artist, title in zip(ids, artists, titles):
        # print(f"{id} {artist} {title}")
        link = scrape_link(f'{artist}, {title}')
        try:
            filename = download_video(link)
            save_index_name_and_link('downloaded.txt', id, title, link)
            convert_to_mp3_and_change_name(filename, id)
        except:
            save_index_name_and_link('downloaded.txt', id, title, link, opt="failed")

    remove_all_mp4_files("./")
    print("DONE")
    