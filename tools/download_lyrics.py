import os
import json
import pandas as pd

from lyricsgenius import Genius 
from requests.exceptions import HTTPError, Timeout
from http import HTTPStatus

from download_songs import read_config, read_excel_database


ILLEGAL_FILENAME_CHARAKTERS = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "[", "]", "{", "}", ";", 
                               ":", "=", ".", "/", "<", ">", "?", "|", "`", "~", "-", "=", "+", "\""]

def authorize_genius(config):
    try:
        genius = Genius(config['access_token'])
        return genius
    except:
        print("Error with authetication")
        exit(1)


def download_lyric(genius, title, artist):
    try:
        downloaded_song = genius.search_song(title = title, artist = artist, get_full_info = True)
        return downloaded_song, HTTPStatus.OK.value
    except HTTPError as e:
        print(e.args[1])
        return "", e.errno
    except Timeout as t:
        print("Request timed out")
        return "", t


def replace_illegal_chars(input_string):
    for illegal_char in ILLEGAL_FILENAME_CHARAKTERS:
        input_string = input_string.replace(illegal_char, "-")
    return input_string


def save_lyric(id, artist, title, mood, song):
    filename =  f"{id}_{artist}_{title}"
    filename = replace_illegal_chars(filename)
    
    song = {
        'id': id,
        'mood': mood,
        'artist': artist,
        'title': title,
        'song': song
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
            downloaded_song, status_code = download_lyric(genius, title, artist)
            if status_code == HTTPStatus.OK.value:
                try:
                    song = downloaded_song.to_dict()
                except:
                    song = {}
                    print("downloaded_song is empty")
                    log_error(id, artist, title, 'empty')
                save_lyric(id, artist, title, mood, song)
            else:
                log_error(id, artist, title, status_code)


if __name__ == '__main__':
    START_ID = 786
    main(START_ID)
    