import os
import re
import random
import fasttext
import pandas as pd

from datetime import datetime
from csv import QUOTE_NONE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# from tools.extract_features_from_lyric import load_en_dataset, clean_lyric
from extract_features_from_lyric import load_en_dataset, clean_lyric


SEED = 100

def fasttext_preprocess(dataset, save_path, remove_newline=False):

    for index, row in dataset.iterrows():
        lyric, _ = clean_lyric(row['lyric'], row['title'])
        
        if remove_newline:
            lyric = lyric.replace('\n', ' ')

        dataset.at[index, 'lyric'] = lyric
        dataset.at[index, 'mood'] = '__label__' + row['mood'] 
    
    dataset = dataset[['mood', 'lyric']]

    dataset[['mood', 'lyric']].to_csv(save_path, 
                                      sep=' ', 
                                      index=False, 
                                      header=False,
                                      quoting=QUOTE_NONE,
                                      quotechar="",
                                      escapechar=" ")

    return dataset

def train_fasttext(hyperparams):

    model = fasttext.train_supervised(input=hyperparams['train'], 
                                     autotuneValidationFile=hyperparams['valid'],
                                     autotuneDuration=12)

    now = datetime.now()
    path_to_model = os.path.join('..', 'models', 'lyric', 'fasttext_models', f'fasttext_model_{now.strftime("%d%m%Y_%H%M%S")}.bin')
    model.save_model(path_to_model) 

    print(f'\nModel best parameters: \n'
          f'size of the context window: {model.ws} \n'
          f'wordNgrams: {model.wordNgrams} \n'
          f'loss function: {model.loss} \n'
          f'learning rate: {model.lr} \n'
          f'number of epochs: {model.epoch} \n'
          )

    _, precision, recall = model.test(hyperparams['test'])

    print(f'Test set precision: {precision}')
    print(f'Test set recall: {recall}')
    print(f'Test set F1-score: {2 * (precision * recall) / (precision + recall)}')


    return path_to_model  

def test_fasttext(test_path, path_to_model, remove_newline):

    model = fasttext.load_model(path_to_model)

    labels = {'__label__angry': 0, '__label__happy': 1, '__label__relaxed': 2, '__label__sad': 3}

    y_true = []
    y_pred = []

    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            if remove_newline:
                y_true.append(labels[line.split(' ')[0]])
                line = re.sub(f'__label__{y_true} ', '', line)
                line = re.sub('\n', ' ', line)
                score = model.predict([str(line)])
                y_pred.append(labels[score[0][0][0]]) 
            else:

                #get secont approach 
                #have to concatenate all lines per one label and then predict
                
                try:
                    y_true.append(labels[line.split(' ')[0]])
                except KeyError:
                    continue  
            
            

    target_names = ['angry', 'happy', 'relaxed', 'sad']
    print(classification_report(y_true, y_pred, target_names=target_names))

def main():

    dataset_path = os.path.join('..', '..', 'database', 'lyrics')
    duplicate_path = os.path.join('..', 'database', 'removed_rows.json') 

    en_dataset = load_en_dataset(dataset_path, duplicate_path)

    train, test = train_test_split(en_dataset, test_size=0.3, random_state=SEED)
    valid, test = train_test_split(test, test_size=0.5, random_state=SEED)

    remove_newline = True

    output_path_train = os.path.join('..', 'database', 'fasttext', 'lyric.train')
    if not os.path.exists(output_path_train):
        fasttext_preprocess(train, output_path_train, remove_newline)
    
    output_path_valid = os.path.join('..', 'database', 'fasttext', 'lyric.test')
    if not os.path.exists(output_path_valid):
        fasttext_preprocess(valid, output_path_valid, remove_newline)

    output_path_test = os.path.join('..', 'database', 'fasttext', 'lyric.valid')
    if not os.path.exists(output_path_test):
        fasttext_preprocess(test, output_path_test, remove_newline)
    
    hyperparams = {'train': output_path_train,
                   'valid': output_path_valid, 
                   'test': output_path_test}

    path_to_model = train_fasttext(hyperparams)
    
    test_fasttext(hyperparams['test'], path_to_model, remove_newline)  
     


if __name__ == '__main__':
    main()