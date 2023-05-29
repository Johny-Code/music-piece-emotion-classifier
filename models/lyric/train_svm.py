import os

from tools.extract_features_from_lyric import load_en_dataset

def main():
    dataset_path = os.path.join('..', '..', 'database', 'lyrics')
    duplicate_path = os.path.join('..', 'database', 'removed_rows.json') 

    en_dataset = load_en_dataset(dataset_path, duplicate_path)
    
    print(en_dataset.head())

if __name__ == '__main__':
    main()