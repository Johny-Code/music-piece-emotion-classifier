import os
import sys
import time
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.preprocessing import StandardScaler

SEED = 100
TARGET_NAMES = ['happy', 'angry', 'sad', 'relaxed']


def read_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df


def train_svm(svm_params, X_train, y_train, X_test, y_test):

    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED)

    print(f'\nSVM parameters: \n'
          f'kernel: {svm_params["kernel"]} \n'
          f'gamma: {svm_params["gamma"]} \n'
          )

    start = time.time()

    svm_clf = svm.SVC(kernel=svm_params['kernel'], gamma=svm_params['gamma'], C=svm_params['C'])
    svm_clf.fit(X_train, y_train)
    end = time.time()
    print(f'Training time: {end - start}')

    start = time.time()
    y_pred = svm_clf.predict(X_test)
    end = time.time()
    print(f'Prediction time: {end - start}')

    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return svm_clf


def grid_search_svm(X_train, y_train, X_test, y_test):

    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED)

    X_train = X_train + X_valid
    y_train = y_train + y_valid

    print(len(X_train))
    print(len(y_train))

    print('Grid search for SVM')

    linear_kernel_params = {
        'C': [ 0.01, 0.1, 1, 10, 100 ], 
        'gamma': [ 0.01, 0.05, 0.1, 0.3, 0.5, 1 ]
    }

    poly_kernel_params = {
        'degree': [ 2, 3, 4, 5, 6 ], 
        'gamma': [ 0.01, 0.05, 0.1, 0.3, 0.5, 1 ], 
        'C': [ 0.01, 0.1, 1, 10, 100 ]
    }

    rbf_kernel_params = {
        'gamma': [ 0.01, 0.05, 0.1, 0.3, 0.5, 1 ], 
        'C': [ 0.01, 0.1, 1, 10, 100 ]
    }

    kernel_sigmoid_params = {
        'gamma': [ 0.01, 0.05, 0.1, 0.3, 0.5, 1 ], 
        'C': [ 0.01, 0.1, 1, 10, 100 ]
    }


    k_fold = 10

    print('Linear kernel')
    for C in linear_kernel_params['C']:
        for gamma in linear_kernel_params['gamma']:
            print('\n *********************************** \n')
            print(f'C: {C}, gamma: {gamma}')
            svm_clf = svm.SVC(kernel='linear', C=C, gamma=gamma)
            scores = cross_val_score(svm_clf, X_train, y_train, cv=k_fold, scoring='accuracy', n_jobs=-1)
            print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})')

    print('Polynomial kernel')
    for degree in poly_kernel_params['degree']:
        for C in poly_kernel_params['C']:
            for gamma in poly_kernel_params['gamma']:
                print('\n *********************************** \n')
                print(f'degree: {degree}, C: {C}, gamma: {gamma}')
                svm_clf = svm.SVC(kernel='poly', degree=degree, C=C, gamma=gamma)
                scores = cross_val_score(svm_clf, X_train, y_train, cv=k_fold, scoring='accuracy', n_jobs=-1)
                print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})')
            
    print('RBF kernel')
    for C in rbf_kernel_params['C']:
        for gamma in rbf_kernel_params['gamma']:
            print('\n *********************************** \n')
            print(f'C: {C}, gamma: {gamma}')
            svm_clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
            scores = cross_val_score(svm_clf, X_train, y_train, cv=k_fold, scoring='accuracy', n_jobs=-1)
            print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})')


    print('Sigmoid kernel')
    for C in kernel_sigmoid_params['C']:
        for gamma in kernel_sigmoid_params['gamma']:
            print('\n *********************************** \n')
            print(f'C: {C}, gamma: {gamma}')
            svm_clf = svm.SVC(kernel='sigmoid', C=C, gamma=gamma)
            scores = cross_val_score(svm_clf, X_train, y_train, cv=k_fold, scoring='accuracy', n_jobs=-1)
            print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})')

def clean_features(df):

    target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}

    X = []
    y = []

    for row in df.iterrows():
        
        emotion = row[1][0]
        y.append(target_dict[emotion])

        vector = row[1][1]
        vector = vector.replace('[', '')
        vector = vector.replace(']', '')
        vector = vector.replace('\n', '')
        vector = vector.replace('   ', ' ')
        vector = vector.split(' ')

        vector_cleaned = []
        for value in vector:
            if value != '':
                vector_cleaned.append(float(value))

        temp = []
        temp += vector_cleaned
        for i, ele in enumerate(row[1][2:11]):
            if i < 10:
                if ele == 'True':
                    temp.append(1)
                elif ele == 'False':
                    temp.append(0)
                temp.append(ele)

        X.append(temp)

    return X, y


def load_data(scaling=False):

    input_path = os.path.join('database', 'features', 'lyric_features.csv')
    df = read_data(input_path)

    X, y = clean_features(df)

    if scaling:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_run', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    args = parser.parse_args()

    if args.simple_run:
        X_train, X_test, y_train, y_test = load_data()
        svm_params = {'kernel': 'linear', 'gamma': 0.1, 'C': 0.01}
        _ = train_svm(svm_params, X_train, y_train, X_test, y_test)

    elif args.grid_search:
        X_train, X_test, y_train, y_test = load_data()
        grid_search_svm(X_train, y_train, X_test, y_test)

    else:
        print('Please specify --simple_run or --grid_search')
        print('For simple run: python train_svm.py --simple_run')
        print('For grid search: python train_svm.py --grid_search')
        sys.exit(0)
