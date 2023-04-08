from moviepy.editor import *


def convert_to_mp3_and_change_name(filename):
    video = VideoFileClip(filename)
    video.audio.write_audiofile(f"../database/songs/{file[:-3]}mp3")


if __name__=="__main__":
    for file in os.listdir("../database/to_be_converted"):
        if file.endswith(".mp4"):
            convert_to_mp3_and_change_name(f"../database/to_be_converted/{file}")
