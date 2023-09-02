import os
import sys
import argparse
import torch
import time
import datetime
import math
import numpy as np
import pandas as pd
from transformers import XLNetTokenizer, XLNetModel, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

sys.path += ['tools/', '../../tools/', 'implementation/dataset', 'implementation']
from CustomLyricTensorDataset import CustomLyricTensorDataset
from XLNetForMultiLabelSequenceClassification import XLNetForMultiLabelSequenceClassification
from extract_features_from_lyric import load_en_dataset, clean_lyric


SEED = 100
torch.manual_seed(SEED)
TARGET_NAMES = ['happy', 'angry', 'sad', 'relaxed']


def preprocess(dataset, remove_newline):
    target_dict = {'happy': [1.,0.,0.,0.], 'angry': [0.,1.,0.,0.], 'sad': [0.,0.,1.,0.], 'relaxed': [0.,0.,0.,1.]}

    for index, row in dataset.iterrows():
        lyric, _ = clean_lyric(row['lyric'], row['title'])
        if remove_newline:
            lyric = lyric.replace('\n', ' ')
        
        dataset.at[index, 'lyric'] = lyric
        dataset.at[index, 'mood'] = target_dict[dataset.at[index, 'mood']]

    dataset = dataset[['mood', 'lyric', 'split']]
    train_dataset = dataset[dataset['split'] == 'train']
    test_dataset = dataset[dataset['split'] == 'test']
    val_dataset = dataset[dataset['split'] == 'val']

    return train_dataset, test_dataset, val_dataset


def load_dataset(dataset_path, database_path):
    en_dataset = load_en_dataset(dataset_path, database_path)
    remove_newline = True
    train_dataset, test_dataset, val_dataset = preprocess(en_dataset, remove_newline)
    return train_dataset, test_dataset, val_dataset


def tokenize_inputs(hyperparameters, lyrics, tokenizer):
    tokenized_texts = []
    input_ids = []
    for lyric in lyrics:
        tokens = tokenizer.tokenize(lyric)[:hyperparameters['tokenizer']['num_embeddings']-2]
        tokenized_texts.append(tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        
        if len(ids) < hyperparameters['tokenizer']['num_embeddings']:
            delta_len = hyperparameters['tokenizer']['num_embeddings'] - len(ids)
            ids = ids + [0] * delta_len    

        input_ids.append(ids)

    return np.array(input_ids)


def create_attention_masks(input_ids):
    attention_mask = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_mask.append(seq_mask)

    return np.array(attention_mask)


def to_tensor(input_ids, attention_masks, labels):
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels


def to_tensorDataset_dataLoader_tuple(input_ids, attention_masks, labels, hyperparameters):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), 
                            batch_size=hyperparameters['model']['batch_size'])
    
    return dataset, dataloader


def to_custom_tensorDataset_dataLoader_tuple(input_ids, attention_masks, labels, hyperparameters, labels_names=None):
    dataset = CustomLyricTensorDataset(input_ids, attention_masks, labels, labels_names=labels_names)

    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), 
                            batch_size=hyperparameters['model']['batch_size'])
    
    return dataset, dataloader


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_model(model, output_dir):
    now = datetime.datetime.now()
    file_name = "xlnet_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".pt"
    output_path = os.path.join(output_dir, file_name)
    torch.save(model.state_dict(), output_path)
    print(f'Model saved to {output_path}')


def load_model(path_to_model):
    model = XLNetForMultiLabelSequenceClassification()
    model.load_state_dict(torch.load(path_to_model))
    return model


def train(model, optimizer, train_dataloader, validation_dataloader, hyperparameters, train_loss_set = [], valid_loss_set = [],
          start_epoch = 0, lowest_eval_loss = None):

    training_stats = []
    total_t0 = time.time()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for i in range(hyperparameters['model']['epochs']):
        # if continue training from saved model
        actual_epoch = start_epoch + i
        print(f"\nEpoch {actual_epoch}/{hyperparameters['model']['epochs']}")
        print('Training...')

        t0 = time.time()
        
        model.train()
        
        tr_loss = 0
        num_train_samples = 0

        for step, batch in enumerate(train_dataloader):
            if step % 3 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.')

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad() # Clear out the gradients (by default they accumulate)
            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels) # Forward pass
            tr_loss += loss.item()
            
            num_train_samples += b_labels.size(0)

            loss.backward() # Backward pass
            optimizer.step() # Update parameters and take a step using the computed gradient

        epoch_train_loss = tr_loss/num_train_samples
        train_loss_set.append(epoch_train_loss)
        training_time = format_time(time.time() - t0)

        print(f"\n  Average training loss: {round(epoch_train_loss, 3)}")
        print(f"  Training epcoh took: {training_time} \n")
        print("Running Validation...")
        
        t0 = time.time()
        
        model.eval()
        
        eval_loss = 0
        num_eval_samples = 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)
        
        epoch_eval_loss = eval_loss/num_eval_samples
        valid_loss_set.append(epoch_eval_loss)
        validation_time = format_time(time.time() - t0)

        print(f"  Validation Loss: {epoch_eval_loss}")
        print(f"  Validation took: {validation_time}")

        training_stats.append(
            {
                'epoch': actual_epoch,
                'Training Loss': epoch_train_loss,
                'Valid. Loss': epoch_eval_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
            print(f'Best performance achived at epoch {actual_epoch} with validation loss of {lowest_eval_loss}')
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
                print(f'At epoch {actual_epoch} better performance was achived with validation loss of {lowest_eval_loss}')
    
    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time()-total_t0)}")

    save_model(model, hyperparameters['model_save_path'])

    return model, train_loss_set, valid_loss_set, training_stats


def test_model(model,test_dataloader):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    y_pred = []
    y_true = []
    print("\nRunning Testing...")

    for index, batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            output = model(b_input_ids, attention_mask=b_input_mask)
            output = output.detach().cpu().numpy()

        for predict in np.argmax(output, axis=1):
            y_pred.append(predict)

        labels_ids = b_labels.to('cpu').numpy()    
        for label in np.argmax(labels_ids, axis=1):
            y_true.append(label)

    print("Testing complete!\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=3))


def simple_run(hyperparameters):
    dataset_path = os.path.join('../../', 'database', 'lyrics')
    database_path = os.path.join('../../database', 'MoodyLyrics4Q_cleaned_split.csv')

    train_dataset, test_dataset, val_dataset = load_dataset(dataset_path, database_path)
    
    tokienizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case = hyperparameters['tokenizer']['do_lower_case'])

    train_labels = np.array(train_dataset['mood'].tolist())    
    train_input_ids = tokenize_inputs(hyperparameters, train_dataset['lyric'].tolist(), tokienizer)
    train_attention_masks  = create_attention_masks(train_input_ids)

    test_labels = np.array(test_dataset['mood'].tolist())
    test_input_ids = tokenize_inputs(hyperparameters, test_dataset['lyric'].tolist(), tokienizer)
    test_attention_masks  = create_attention_masks(test_input_ids)

    val_labels = np.array(val_dataset['mood'].tolist())
    val_input_ids = tokenize_inputs(hyperparameters, val_dataset['lyric'].tolist(), tokienizer)
    val_attention_masks  = create_attention_masks(val_input_ids)

    train_input_ids, train_attention_masks, train_labels = to_tensor(train_input_ids, train_attention_masks, train_labels)
    val_input_ids, val_attention_masks, val_labels = to_tensor(val_input_ids, val_attention_masks, val_labels)
    test_input_ids, test_attention_masks, test_labels = to_tensor(test_input_ids, test_attention_masks, test_labels)
            
    train_dataset, train_dataloader = to_tensorDataset_dataLoader_tuple(train_input_ids, train_attention_masks, train_labels, hyperparameters)
    val_dataset, val_dataloader = to_tensorDataset_dataLoader_tuple(val_input_ids, val_attention_masks, val_labels, hyperparameters)
    test_dataset, test_dataloader = to_custom_tensorDataset_dataLoader_tuple(test_input_ids, test_attention_masks, test_labels, hyperparameters,
                                                              labels_names=test_dataset['mood'].index.values.tolist())

    model = XLNetForMultiLabelSequenceClassification(num_labels=hyperparameters['model']['num_labels'])

    optimizer = AdamW(model.parameters(), 
                      lr=hyperparameters['model']['lr'], 
                      weight_decay=hyperparameters['model']['weight_decay'],
                      correct_bias=hyperparameters['model']['correct_bias'], 
                      )
    
    model, train_loss_set, valid_loss_set, training_stats = train(model, optimizer, train_dataloader, val_dataloader, hyperparameters)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    print(df_stats)

    test_model(model, test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--test_model', action='store_true')
    parser.add_argument('--grid_search', action='store_true')
    args = parser.parse_args()

    if args.fine_tune:     
        #actual date and time
        model_save_folder = os.path.join('models', 'lyric', 'xlnet')
        os.makedirs(model_save_folder, exist_ok=True)
        hyperparameters = {
                            'tokenizer':{
                                'do_lower_case': False,
                                'num_embeddings': 256,
                            },
                            'model':{
                                'num_labels': 4,
                                'batch_size': 64, #sould be 32
                                'lr': 2e-5,
                                'weight_decay': 0.01,
                                'correct_bias': False,
                                'epochs': 4,
                            },
                            'model_save_path': model_save_folder
                        }
        simple_run(hyperparameters)
    
    elif args.test_model:
        hyperparameters = {
                            'tokenizer':{
                                'do_lower_case': False,
                                'num_embeddings': 128,
                            },
                            'model':{
                                'num_labels': 4,
                                'batch_size': 32,
                            }
                            }
        
        path_to_model = os.path.join('models', 'lyric', 'xlnet', 'xlnet_2023-09-02_10-41-03.pt')
        model = load_model(path_to_model)
        
        dataset_path = os.path.join('../../', 'database', 'lyrics')
        database_path = os.path.join('../../database', 'MoodyLyrics4Q_cleaned_split.csv')

        _, test_dataset, _ = load_dataset(dataset_path, database_path)

        tokienizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case = hyperparameters['tokenizer']['do_lower_case'])

        test_labels = np.array(test_dataset['mood'].tolist())
        test_input_ids = tokenize_inputs(hyperparameters, test_dataset['lyric'].tolist(), tokienizer)
        test_attention_masks  = create_attention_masks(test_input_ids)

        test_input_ids, test_attention_masks, test_labels = to_tensor(test_input_ids, test_attention_masks, test_labels)
        
        test_dataset, test_dataloader = to_custom_tensorDataset_dataLoader_tuple(test_input_ids,
                                                                                 test_attention_masks,
                                                                                 test_labels, 
                                                                                 hyperparameters,
                                                                                 test_dataset['mood'].index.values.tolist())

        test_model(model,test_dataloader)
    
    elif args.grid_search:
        do_lower_case = [True, False]
        num_embeddings = [128, 256, 512]
        batch_sizes = [32, 64, 128]
        iteration = 0
        for lower_case in do_lower_case:
            for num_embedding in num_embeddings:
                for batch_size in batch_sizes:  
                    print(iteration)
                    iteration += 1
                    hyperparameters = {
                                        'tokenizer':{
                                            'do_lower_case': lower_case,
                                            'num_embeddings': num_embedding,
                                        },
                                        'model':{
                                            'num_labels': 4,
                                            'batch_size': batch_size,
                                            'lr': 2e-5,
                                            'weight_decay': 0.01,
                                            'correct_bias': False,
                                            'epochs': 5,
                                        },
                                    }

                    simple_run(hyperparameters)

                    for key, value in hyperparameters.items():
                        print(f"{key} : {value}")

                    print('***************************************************\n\n')
                    
    else:
        print('No arguments passed. Use --fine_tune to train model, --test_model to test model or --grid_search to perform grid search')  
        exit(1)
        