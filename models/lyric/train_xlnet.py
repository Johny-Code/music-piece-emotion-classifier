import argparse
import sys

sys.path.append('tools/')
from extract_features_from_lyric import load_en_dataset, clean_lyric

def preprocess(dataset, remove_newline):

    target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}

    for index, row in dataset.iterrows():
        lyric, _ = clean_lyric(row['lyric'], row['title'])

        if remove_newline:
            lyric = lyric.replace('\n', ' ')
        
        dataset.at[index, 'lyric'] = lyric
        dataset.at[index, 'target'] = target_dict[dataset.at[index, 'mood']]

    dataset = dataset[['mood', 'lyric']]

    print(dataset.head())

    return dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_run', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    args = parser.parse_args()

    if args.simple_run:
        
        en_dataset = load_en_dataset()

        remove_newline = True
        dataset = preprocess(en_dataset, remove_newline)
        
        pass

    elif args.grid_search:
        pass

    else:
        print('Please specify --simple_run or --grid_search')
        print('For simple run: python train_svm.py --simple_run')
        print('For grid search: python train_svm.py --grid_search')
        sys.exit(0)