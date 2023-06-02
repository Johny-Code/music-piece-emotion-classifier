import sys
import argparse
import keras
import os
import wandb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from wandb.keras import WandbMetricsLogger

from train_svm import read_data, TARGET_NAMES, SEED

def clean_features(df):

    target_dict = {'happy': [1., 0., 0., 0.], 
                   'angry': [0., 1., 0., 0.],
                    'sad':  [0., 0., 1., 0.],
                    'relaxed': [0., 0., 0., 1.]}

    X = []
    y = []

    for row in df.iterrows():

        emotion = row[1][0]
        y.append(target_dict[emotion])    

        vector = row[1][1]
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

    return np.array(X), np.array(y)

def load_data():

    input_path = os.path.join('database', 'features', 'lyric_features.csv')
    df = read_data(input_path)

    X, y = clean_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    return X_train, X_test, y_train, y_test

def build_4_dense_ann(input_size=309, dense_size=128, output_size=4, activation='relu', dropout=0.2, optimizer='adam'):

    model = keras.models.Sequential([
        keras.layers.Dense(dense_size, input_dim=input_size, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(dense_size, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(dense_size, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(dense_size, activation=activation),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(output_size, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train_ann(X_train, y_train, X_test, y_test, params):

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=SEED)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    input_size = 309
    output_size = 4

    model = build_4_dense_ann(input_size, params['dense_size'], output_size, params['activation'], params['dropout'], params['optimizer'])

    _ = model.fit(X_train, y_train, 
                        epochs=params['epochs'], 
                        batch_size=params['batch_size'], 
                        validation_data=(X_val, y_val),
                        callbacks=[WandbMetricsLogger()])

    score = model.evaluate(X_test, y_test, batch_size=params['batch_size'])

    print(f'Test accuracy: {score[1]}')

    y_pred = model.predict(X_test)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)
    wandb.log({"conf_mat": cm})

    report = classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=TARGET_NAMES, output_dict=True)
    print(report)

    wandb.log({"raw_report": report})

    wandb.log({"happy_precision":  report['happy']['precision'],
                "happy_recall":     report['happy']['recall'],
                "happy_f1":         report['happy']['f1-score'],
                "happy_support":    report['happy']['support'],
                "angry_precision":  report['angry']['precision'],
                "angry_recall":     report['angry']['recall'],
                "angry_f1":         report['angry']['f1-score'],
                "angry_support":    report['angry']['support'],
                "sad_precision":    report['sad']['precision'],
                "sad_recall":       report['sad']['recall'],
                "sad_f1":           report['sad']['f1-score'],
                "sad_support":      report['sad']['support'],
                "relaxed_precision":  report['relaxed']['precision'],
                "relaxed_recall":     report['relaxed']['recall'],
                "relaxed_f1":         report['relaxed']['f1-score'],
                "relaxed_support":    report['relaxed']['support']})
    
    wandb.log({"macro_precision":  report['macro avg']['precision'],
                "macro_recall":     report['macro avg']['recall'],
                "macro_f1":         report['macro avg']['f1-score'],
                "macro_support":    report['macro avg']['support'],
                "weighted_precision":  report['weighted avg']['precision'],
                "weighted_recall":     report['weighted avg']['recall'],
                "weighted_f1":         report['weighted avg']['f1-score']})

def simple_run(config):

    wandb.init(project='feature-based-4-dense-ann',
                config=config)
                
    X_train, X_test, y_train, y_test = load_data()

    train_ann(X_train, y_train, X_test, y_test, config)

    wandb.finish()

def grid_search_ann():
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--simple_run', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    args = parser.parse_args()

    if args.simple_run:
        config={'lr': 0.02,
                'epochs': 100,
                'batch_size': 32,
                'dense_size': 128,
                'activation': 'relu',
                'dropout': 0.2,
                'optimizer': 'adam'
                }
        
        simple_run(config)
        

    elif args.grid_search:
        pass

    else:
        print('Please specify a flag.')
        print('For simple run: python train_ann.py --simple_run')
        print('For grid search: python train_ann.py --grid_search')
        sys.exit(0)
