import os
import sys
import time
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

from utils.draw_plot import draw_confusion_matrix


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
    draw_confusion_matrix(cm, TARGET_NAMES)

    return svm_clf

def grid_search_svm(X_train, y_train, X_test, y_test):
    
    params = [
    { 'kernel': ['linear'], 'C': [0.001, 0.01, 1, 10, 100]},
    { 'kernel': ['rbf', 'sigmoid'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
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
    draw_confusion_matrix(cm, TARGET_NAMES, 'svm_best_grid_search.png')


def main():
    
    input_path = os.path.join('..', '..','database', 'features', 'lyric_features.csv')
    df = read_data(input_path)

    target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}
    title_in_lyric_dict = {'True': 1, 'False': 0}

    df.replace({"emotion": target_dict}, inplace=True)
    df.replace({"title_in_lyric": title_in_lyric_dict}, inplace=True)

    df['lyrics_vector'] = df['lyrics_vector'].apply(lambda x: x[1:-1].split(','))

    y = df['emotion']
    X = df.drop(['emotion'], axis=1)

    #single experiment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = SEED)

    svm_params = {'kernel': 'rbf', 'gamma': 0.3}
    
    _ = train_svm(svm_params, X_train, y_train, X_test, y_test)

    #grid search
    grid_search_svm(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()