import os
import re
import fasttext
import pandas as pd

READABLE_LABELS = {'angry': 0, 'happy': 1, 'relaxed': 2, 'sad': 3}
LABELS = {'__label__angry': 0, '__label__happy': 1, '__label__relaxed': 2, '__label__sad': 3}


def main():
    path_to_model = os.path.join('fasttext_models', 'fasttext_model.bin')

    model = fasttext.load_model(path_to_model)

    test_path = os.path.join('..', '..', 'database', 'fasttext', 'lyric.test')

    y_true = []
    y_pred = []

    #read test data line by line 
    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            #get first word of line
            try:
                y_true.append(LABELS[line.split(' ')[0]])
            except KeyError:
                continue  
            line = re.sub(f'__label__{y_true} ', '', line)
            line = re.sub('\n', ' ', line)
            score = model.predict([str(line)])
            y_pred.append(LABELS[score[0][0][0]]) 

            if i == 10:
                break
            i += 1

    print(y_true)
    print(y_pred)
            



if __name__ == '__main__':
    main()