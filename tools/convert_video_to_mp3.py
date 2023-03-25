from moviepy.editor import *


def convert_to_mp3_and_change_name(filename):
    video = VideoFileClip(filename)
    video.audio.write_audiofile(f"../database/{file[:-3]}.mp3")


if __name__=="__main__":
    for file in os.listdir("../database"):
        if file.endswith(".mp4"):
            convert_to_mp3_and_change_name(f"../database/{file}")
