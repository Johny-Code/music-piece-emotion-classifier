import sys
import argparse
import keras
import os
import wandb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from wandb.keras import WandbMetricsLogger

from train_svm import read_data, TARGET_NAMES, SEED
# from utils.draw_plot import draw_confusion_matrix, plot_acc_loss

def clean_features(df):

    target_dict = {'happy': [1., 0., 0., 0.], 
                   'angry': [0., 1., 0., 0.],
                    'sad':  [0., 0., 1., 0.],
                    'relaxed': [0., 0., 0., 1.]}

    ohe = OneHotEncoder( categories='auto',  # Categories per feature
            drop=None, # Whether to drop one of the features
            sparse=True, # Will return sparse matrix if set True
            handle_unknown='error' # Whether to raise an error 
        ) 

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

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=SEED)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    input_size = 309
    output_size = 4

    model = build_4_dense_ann(input_size, params['dense_size'], output_size, params['activation'], params['dropout'], params['optimizer'])

    history = model.fit(X_train, y_train, 
                        epochs=params['epochs'], 
                        batch_size=params['batch_size'], 
                        validation_data=(X_val, y_val),
                        callbacks=[WandbMetricsLogger()])

    # acc_loss_out_path = os.path.join('models', 'lyric', 'ann_acc_loss.png')
    # plot_acc_loss(history, acc_loss_out_path)

    score = model.evaluate(X_test, y_test, batch_size=params['batch_size'])

    print(f'Test accuracy: {score[1]}')

    y_pred = model.predict(X_test)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cm)

    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=TARGET_NAMES))

    wandb.log(f"Classification report {classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=TARGET_NAMES)}")

    # output_path = os.path.join('models', 'lyric', 'history', 'ann')
    # draw_confusion_matrix(cm, TARGET_NAMES, )


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
        
        run = wandb.init(project='feature-based-ann',
                   
                   config=config
                    
                )
                    
        X_train, X_test, y_train, y_test = load_data()

        train_ann(X_train, y_train, X_test, y_test, config)

        wandb.finish()

    elif args.grid_search:
        pass
    else:
        print('Please specify a flag.')
        print('For simple run: python train_ann.py --simple_run')
        print('For grid search: python train_ann.py --grid_search')
        sys.exit(0)
