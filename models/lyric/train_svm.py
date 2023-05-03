import os
import pandas as pd

def read_data(filepath):
    df = pd.read_csv(filepath, index_col = 0)
    return df

def main():
    path_to_database = os.path.join('..', '..', 'database', 'features.csv')
    df = read_data(path_to_database)

    


if __name__ == '__main__':
    main()