import os
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import svm, metrics

def read_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df

def main():
    
    input_path = os.path.join('../../database/lyrics_features/ML4Q_english_features.csv')
    df = read_data(input_path)

    target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}
    title_in_lyric_dict = {'True': 1, 'False': 0}

    df.replace({"emotion": target_dict}, inplace=True)
    df.replace({"title_in_lyric": title_in_lyric_dict}, inplace=True)

    df['lyrics_vector'] = df['lyrics_vector'].apply(lambda x: x[1:-1].split(','))

    y = df['emotion']
    X = df.drop(['emotion'], axis=1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=5)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    print(X_train.head())

    svm_clf = svm.SVC(kernel='rbf', gamma=0.3)
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)
    
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

if __name__ == '__main__':
    main()