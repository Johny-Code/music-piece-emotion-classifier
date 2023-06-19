import sys
import os
import argparse
import fasttext
import time
import pandas as pd

from datetime import datetime
from csv import QUOTE_NONE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from train_svm import TARGET_NAMES
sys.path.append("utils/")
from draw_plot import draw_confusion_matrix
sys.path.append("tools/")
from extract_features_from_lyric import load_en_dataset, clean_lyric


SEED = 100


def fasttext_preprocess(dataset, save_path, replace_newline):

    for index, row in dataset.iterrows():
        lyric, _ = clean_lyric(row['lyric'], row['title'])

        if replace_newline:
            lyric = lyric.replace('\n', replace_newline)

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

    start = time.time()
    if hyperparams['autotune_duration'] == None:
        model = fasttext.train_supervised(input=hyperparams['train'],
                                            wordNgrams=hyperparams['wordNgrams'],
                                            lr=hyperparams['lr'],
                                            ws=hyperparams['ws'],  # size of the context window
                                            epoch=hyperparams['epoch'],
                                            loss=hyperparams['loss'],
                                            thread=hyperparams['thread'])
    else:
        model = fasttext.train_supervised(input=hyperparams['train'], 
                                          autotuneValidationFile=hyperparams['valid'], 
                                          autotuneDuration=hyperparams['autotune_duration'],
                                          loss = hyperparams['loss'])

    end = time.time()
    print(f'Training time: {round((end - start), 2)} seconds')

    print(f'\nModel best parameters: \n'
          f'size of the context window: {model.ws} \n'
          f'wordNgrams: {model.wordNgrams} \n'
          f'loss function: {model.loss} \n'
          f'learning rate: {model.lr} \n'
          f'number of epochs: {model.epoch} \n'
          )

    _, precision, recall = model.test(hyperparams['valid'])

    print(f'Test set precision: {precision}')
    print(f'Test set recall: {recall}')
    print(f'Test set F1-score: {2 * (precision * recall) / (precision + recall)}')

    return model


def test_fasttext(test_dataset, model):

    labels = {'__label__angry': 0, '__label__happy': 1, '__label__sad': 2, '__label__relaxed': 3}

    y_true = []
    y_pred = []

    test = []
    for lyric in test_dataset['lyric'].tolist():
        test.append(lyric)

    start = time.time()

    score = model.predict(test)
    i = 0
    for pred_label, true_label in zip(score[0], test_dataset['mood'].tolist()):
        y_true.append(labels[true_label])
        y_pred.append(labels[pred_label[0]])

    end = time.time()

    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # output_path = os.path.join('models', 'lyric', 'history', 'svm')
    # draw_confusion_matrix(cm, TARGET_NAMES, output_path)

    print(f'Testing time: {round((end - start), 2)} seconds')

def create_dataset(replace_newline):
    dataset_path = os.path.join('..', 'database', 'lyrics')
    duplicate_path = os.path.join('database', 'removed_rows.json')

    en_dataset = load_en_dataset(dataset_path, duplicate_path)

    train, test = train_test_split(en_dataset, test_size=0.3, random_state=SEED)

    test, valid = train_test_split(test, test_size=0.5, random_state=SEED)

    dataset_path = os.path.join('database', 'fasttext')
    os.makedirs(os.path.join(dataset_path), exist_ok=True)

    output_path_train = os.path.join(dataset_path, 'lyric.train')
    train_dataset = fasttext_preprocess(train, output_path_train, replace_newline)

    output_path_valid = os.path.join(dataset_path, 'lyric.valid')
    valid_dataset = fasttext_preprocess(valid, output_path_valid, replace_newline)

    output_path_test = os.path.join(dataset_path, 'lyric.test')
    test_dataset = fasttext_preprocess(test, output_path_test, replace_newline)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Valid dataset size: {len(valid_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    return output_path_train, output_path_valid, output_path_test, test_dataset

def simple_run(hyperparams):

    start = time.time()
    hyperparams['train'], hyperparams['valid'], hyperparams['test'], test_dataset = create_dataset(hyperparams['replace_newline'])
    end = time.time()
    print(f'Creating dataset took {round((end - start), 2)} seconds')

    model = train_fasttext(hyperparams)

    test_fasttext(test_dataset, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_run', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument('--autotune', action='store_true')
    args = parser.parse_args()

    if args.simple_run:

        hyperparams = {'train': '',
                       'test': '',
                       'valid': '',
                       'wordNgrams': 3,
                       'lr': 0.8,
                       'ws': 5,
                       'epoch': 20,
                       'loss': 'softmax',
                       'thread': 50,
                       'replace_newline': ' ',
                        'autotune_duration': None 
                       }

        simple_run(hyperparams)

    elif args.grid_search:

        params = {'wordNgrams': [1, 2, 3, 4, 5],
                  'lr': [0.1, 0.3, 0.5, 0.8],
                  'epoch': [5, 20, 50]}

        for wordNgrams in params['wordNgrams']:
            for lr in params['lr']:
                for epoch in params['epoch']:
                    hyperparams = {'train': '',
                                    'test': '',
                                    'valid': '',
                                    'wordNgrams': wordNgrams,
                                    'lr': lr,
                                    'ws': 5,
                                    'epoch': epoch,
                                    'loss': 'softmax',
                                    'thread': 50,
                                    'replace_newline': ' ',
                                    'autotune_duration': None
                                    }
                    print('\n**************************************************')
                    for key, value in hyperparams.items():
                        print(f'{key}: {value}')
                              
                    simple_run(hyperparams)
                        
    elif args.autotune:
        hyperparams = {'train': '',
                       'test': '',
                       'valid': '',
                       'wordNgrams': 2,
                       'lr': 0.1,
                       'ws': 5,
                       'epoch': 5,
                       'loss': 'ova',
                       'thread': 4,
                       'replace_newline': ' ',
                        'autotune_duration': 1800
                       }

        simple_run(hyperparams)

    else:
        print('Please specify --simple_run or --grid_search')
        print('For simple run: python train_svm.py --simple_run')
        print('For grid search: python train_svm.py --grid_search')
        sys.exit(0)
