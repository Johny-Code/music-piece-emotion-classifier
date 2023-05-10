import os
import pandas as pd

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def read_clean_featuers(path):
    dataset = pd.read_csv(path, index_col=0)

    labels = {'angry': 0, 'happy': 1, 'relaxed': 2, 'sad': 3}

    y = []
    X = []

    for row in dataset.iterrows():
        y.append(labels[row[0]])

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
                temp.append(ele)
        
        temp.extend(vector_cleaned)

        X.append(temp)
    
    return X, y

def train_svm(X, y):
    k = 10

    clf = svm.SVC(kernel='linear', C=0.01)
    scores = cross_val_score(clf, X, y, cv=k)

    print(scores)

def grid_search(X, y):
    k = 10

    params = [{ 'kernel': ['linear'], 'C': [0.01, 0.05, 1, 10, 100]},
              { 'kernel': ['rbf', 'sigmoid'], 'C': [0.01, 0.05, 0.1, 0.3, 0.8, 1, 3, 10, 50, 100, 150, 200]}]

    gs = GridSearchCV(svm.SVC(), params, cv=k, n_jobs=-1, verbose=False)
    gs.fit(X, y) 

    svm_best = gs.best_estimator_
    best_params = gs.best_params_
    print('Best parameters:', best_params)

    scores = cross_val_score(svm_best, X, y, cv=k)
    print(f"Accuracy for k={k}: {round(scores.mean(), 2)} (+/- {round((scores.std() * 1.96), 2)})")

def main():

    path_to_features = os.path.join('..', '..', 'database', 'features', 'features.csv')

    X, y = read_clean_featuers(path_to_features)

    train_svm(X, y)

    grid_search(X, y)

if __name__ == '__main__':
    main()