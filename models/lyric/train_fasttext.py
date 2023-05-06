import os
import fasttext

from csv import QUOTE_NONE
from sklearn.model_selection import train_test_split

from tools.extract_features_from_lyric import load_en_dataset, clean_lyric

SEED = 100

def fasttext_preprocess(dataset, save_path):

    for index, row in dataset.iterrows():
        lyric, _ = clean_lyric(row['lyric'], row['title'])
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

def train_test_fasttext(hyperparams):

    model = fasttext.train_supervised(input=hyperparams['train'], 
                                     autotuneValidationFile=hyperparams['valid'],
                                     autotuneDuration=600)

    model.save_model(os.path.join('..', 'models', 'lyric', 'fasttext_model.bin')) 

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


    return precision, recall    

def main():

    dataset_path = os.path.join('..', '..', 'database', 'lyrics')
    duplicate_path = os.path.join('..', 'database', 'removed_rows.json') 

    en_dataset = load_en_dataset(dataset_path, duplicate_path)

    train, test = train_test_split(en_dataset, test_size=0.3, random_state=SEED)
    valid, test = train_test_split(test, test_size=0.5, random_state=SEED)


    output_path_train = os.path.join('..', 'database', 'fasttext', 'train.txt')
    if not os.path.exists(output_path_train):
        fasttext_preprocess(train, output_path_train)
    
    output_path_valid = os.path.join('..', 'database', 'fasttext', 'valid.txt')
    if not os.path.exists(output_path_valid):
        fasttext_preprocess(valid, output_path_valid)

    output_path_test = os.path.join('..', 'database', 'fasttext', 'test.txt')
    if not os.path.exists(output_path_test):
        fasttext_preprocess(test, output_path_test)
    
    hyperparams = {'train': output_path_train,
                   'valid': output_path_valid, 
                   'test': output_path_test}

    precision, recall = train_test_fasttext(hyperparams)
    
    #TODO measure accuracy on test set
    
     


if __name__ == '__main__':
    main()
    