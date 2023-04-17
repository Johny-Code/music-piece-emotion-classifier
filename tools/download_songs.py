import pafy 
import googleapiclient.discovery
import pandas as pd
import os
import json
import sys
from moviepy.editor import *


def read_config(*config_files):
    data = {}
    for file in config_files:
        config = open(file)
        data.update(json.load(config))
    return data


def read_excel_database(filename):
    df = pd.read_csv(filename)
    id = df["index"]
    artist = df['artist']
    title = df['title']
    mood = df['mood']
    return id, artist, title, mood


def scrape_link(phrase, config):
    service, version, key = config['api_service_name'], config['api_version'], config['api_key']
    api_service_name = service
    api_version = version
    api_key = key
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
    video.audio.write_audiofile(f"../database/songs/{index}.mp3")
    
    
def remove_all_mp4_files(directory):
    for file in os.listdir(directory):
        if(file.endswith(".mp4")):
            os.remove(file)
    
    
def main(start_id):
    config = read_config("../config/secrets.json", "../config/youtube.json")
    ids, artists, titles, _ = read_excel_database("../database/MoodyLyrics4Q.csv")
    for id, artist, title in zip(ids, artists, titles):
        if int(id[2:]) < start_id:
            continue
        else:
            try:
                link = scrape_link(f'{artist}, {title}', config)
                try:
                    filename = download_video(link)
                    convert_to_mp3_and_change_name(filename, id)
                    save_index_name_and_link('../database/songs/downloaded.txt', id, title, link)
                except UnicodeEncodeError:
                    save_index_name_and_link('../database/songs/downloaded.txt', id, "", "", opt="failed")
                except Exception:
                    save_index_name_and_link('../database/songs/downloaded.txt', id, title, link, opt="failed")
            except googleapiclient.errors.HttpError:
                print("Number of YT API calls has been exceeded.")
                remove_all_mp4_files("./")
                sys.exit(1)
            except Exception:
                print(f"Could not download song {id}")
                save_index_name_and_link('../database/songs/downloaded.txt', id, title, "", opt="failed")
                sys.exit(1)
            
    remove_all_mp4_files("./")
    print("DONE") 
    
    
if __name__=="__main__":
    START_ID = 1903
    main(START_ID)
    