import os

def check_if_all_songs_are_converted():
    real_nb = 0
    
    for file in sorted(os.listdir("../database/songs"), key=len):
        if file.endswith(".mp3"):
            real_nb += 1
            if int(file[2:-4]) != real_nb:
                raise Exception(f"File {int(file[2:-4])-1} is missing")
                
                
if __name__=="__main__":
    check_if_all_songs_are_converted()
