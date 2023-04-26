from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import pandas as pd


def read_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df


if __name__ == "__main__":
    filepath = "../../database/features/1002_stand_norm.csv"
    target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}

    df = read_data(filepath)
    df.replace({"emotion": target_dict}, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.3, random_state=5)

    svm_clf = svm.SVC(kernel='rbf', gamma=0.3)
    # svm_clf = svm.SVC(kernel='poly', degree=2, C=1)
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
