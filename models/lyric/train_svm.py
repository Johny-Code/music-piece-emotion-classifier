import os
import sys
import time
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

sys.path.append("utils/")
from draw_plot import draw_confusion_matrix

SEED = 100
TARGET_NAMES = ['happy', 'angry', 'sad', 'relaxed']


def read_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df


def train_svm(svm_params, X_train, y_train, X_test, y_test):

    print(f'\nSVM parameters: \n'
          f'kernel: {svm_params["kernel"]} \n'
          f'gamma: {svm_params["gamma"]} \n'
          )

    start = time.time()

    svm_clf = svm.SVC(kernel=svm_params['kernel'], gamma=svm_params['gamma'])
    svm_clf.fit(X_train, y_train)
    end = time.time()
    print(f'Training time: {end - start}')

    start = time.time()
    y_pred = svm_clf.predict(X_test)
    end = time.time()
    print(f'Prediction time: {end - start}')

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # output_path = os.path.join('models', 'lyric', 'history', 'svm')

    # draw_confusion_matrix(cm, TARGET_NAMES, output_path)

    return svm_clf


def grid_search_svm(X_train, y_train, X_test, y_test):

    params = [
        {'kernel': ['linear'], 'C': [0.001, 0.01, 1, 10, 100]},
        {'kernel': ['rbf', 'sigmoid'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    ]

    cross_validation = 10
    svm_clf = svm.SVC()

    gs = GridSearchCV(estimator=svm_clf, param_grid=params, cv=cross_validation, scoring='accuracy', verbose=10, n_jobs=10)
    gs.fit(X_train, y_train)

    print(f'Best score: {gs.best_score_}')
    print(f'Best parameters: {gs.best_params_}')

    y_pred = gs.predict(X_test)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    output_path = os.path.join('models', 'lyric', 'history', 'svm')

    draw_confusion_matrix(cm, TARGET_NAMES, output_path)


def clean_features(df):

    target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}

    X = []
    y = []

    for row in df.iterrows():

        print(row[0])
        print(row[1])

        y.append(target_dict[row[0]])

        features = row[1]
        vector = features['lyric_vector']
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
        for i, ele in enumerate(features):
            if i < 10:
                if ele == 'True':
                    temp.append(1)
                elif ele == 'False':
                    temp.append(0)
                temp.append(ele)

        temp.extend(vector_cleaned)

        X.append(temp)

    return X, y


def load_data():

    input_path = os.path.join('database', 'features', 'lyric_features.csv')
    df = read_data(input_path)

    X, y = clean_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_run', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    args = parser.parse_args()

    if args.simple_run:
        X_train, X_test, y_train, y_test = load_data()
        svm_params = {'kernel': 'rbf', 'gamma': 0.3}
        _ = train_svm(svm_params, X_train, y_train, X_test, y_test)

    elif args.grid_search:
        X_train, X_test, y_train, y_test = load_data()
        grid_search_svm(X_train, y_train, X_test, y_test)

    else:
        print('Please specify --simple_run or --grid_search')
        print('For simple run: python train_svm.py --simple_run')
        print('For grid search: python train_svm.py --grid_search')
        sys.exit(0)
