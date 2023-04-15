import os
import json
import re


def clean_lyric(lyric, title):
    
    #remove title and genius annotation
    lyric = re.sub(".+Lyrics.+\]", '',  lyric)

    #removing title (exception)
    lyric = re.sub(f'{title}.+Lyrics', '', lyric)

    #remove exery anotation like [Verse 1], [Chorus], [Bridge], [Part 1] etc.
    lyric = re.sub('\[.+\]', '', lyric)

    #remove every ********* in the lyric
    lyric = re.sub('\*.+\*', '', lyric)

    #remove Genius anotation "You might also like"
    lyric = re.sub('You might also like', '', lyric)

    #remove Embed exist in every lyric in the end
    if lyric[-5:] == 'Embed':
        lyric = re.sub('Embed', '', lyric)
        if lyric[-1].isdigit():
            lyric = re.sub('\d', '', lyric)

    print('-'*50)
    print(lyric)
    print('-'*50)
    return lyric

def main():
    path_dataset = os.path.join('..', '..', 'database', 'lyrics')
    
    for i, file in enumerate(os.listdir(path_dataset)):
        if i == 10:
            break
        if file.endswith('.json'):
            path_file = os.path.join(path_dataset, file)
            with open(path_file, 'r') as f:
                data = json.load(f)
            try:
                lyric = data['song']['lyrics']
                title = data['title']
            except KeyError:
                print(f'KeyError in {path_file}')
                exit(1)

            lyric = clean_lyric(lyric, title)
            print(f'Cleaned {file}')


if __name__ == '__main__':
    main()