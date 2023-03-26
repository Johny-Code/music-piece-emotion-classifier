import os
import json
import pandas as pd

from lyricsgenius import Genius 
from requests.exceptions import HTTPError, Timeout

from download_songs import read_config, read_excel_database
from const import ILLEGAL_FILENAME_CHARAKTERS, STATUS_OK


def authorize_genius(config):
    try:
        genius = Genius(config['access_token'])
        return genius
    except:
        print("Error with authetication")
        exit(1)


def download_lyric(genius, title, artist):
    try:
        downloaded_song = genius.search_song(title = title, artist = artist, get_full_info = False)
        try: 
            lyric = downloaded_song.lyrics
        except:
            lyric = ""
        return lyric, STATUS_OK
    except HTTPError as e:
        print(e.args[1])  # error message
        return "", e.errno
    except Timeout as t:
        print("Request timed out")
        return "", t


def replace_illega_chars(input_string):
    for illegal_char in ILLEGAL_FILENAME_CHARAKTERS:
        input_string = input_string.replace(illegal_char, "-")
    return input_string


def save_lyric(id, artist, title, mood, lyric):
    filename =  f"{id}_{artist}_{title}"
    filename = replace_illega_chars(filename)
    
    song = {
        'id': id,
        'mood': mood,
        'artist': artist,
        'title': title,
        'lyric': lyric
    }

    with open(os.path.join('..', 'database', 'lyrics', f"{filename}.json"), 'w', errors='backslashreplace') as file:
        file.write(json.dumps(song, indent=4))


def log_error(id, artist, title, error):
    with open(os.path.join('.', 'lyric_downloading_errors.log'), 'a') as file:
        file.write(f"For {id}_{artist}_{title} detected error: {error}. \n")


def main(start_id):
    config = read_config(os.path.join('..', 'config', 'genius_secrets.json'))
    ids, artists, titles, mood = read_excel_database(os.path.join('..', 'database', 'MoodyLyrics4Q.csv'))

    genius = authorize_genius(config)

    for id, artist, title, mood in zip(ids, artists, titles, mood):
        if int(id[2:]) < start_id:
            continue
        else:
            lyric, status_code = download_lyric(genius, title, artist)
            if status_code == STATUS_OK:
                if lyric == "":
                    print("lyric is empty")
                    log_error(id, artist, title, 'empty')
                save_lyric(id, artist, title, mood, lyric)
            else:
                log_error(id, artist, title, status_code)


if __name__ == '__main__':
    START_ID = 135
    main(START_ID)
    