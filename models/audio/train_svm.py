from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import pandas as pd


def read_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df


if __name__ == "__main__":
    filepath = "../../database/features/1002_min_max.csv"
    df = read_data(filepath)
    print(df.iloc[:,:])
    # X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test
    # clf = svm.SVC(kernel='rbf') # Linear Kernel
    
    # #Train the model using the training sets
    # clf.fit(X_train, y_train)

    # #Predict the response for test dataset
    # y_pred = clf.predict(X_test)
    
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))