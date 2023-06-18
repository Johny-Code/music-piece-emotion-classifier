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
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

sys.path.append('tools/')
from extract_features_from_lyric import load_en_dataset, clean_lyric


SEED = 100
torch.manual_seed(SEED)

TARGET_NAMES = ['happy', 'angry', 'sad', 'relaxed']

def preprocess(dataset, remove_newline):

    # target_dict = {'happy': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}
    target_dict = {'happy': [1.,0.,0.,0.], 'angry': [0.,1.,0.,0.], 'sad': [0.,0.,1.,0.], 'relaxed': [0.,0.,0.,1.]}


    for index, row in dataset.iterrows():
        lyric, _ = clean_lyric(row['lyric'], row['title'])

        if remove_newline:
            lyric = lyric.replace('\n', ' ')
        
        dataset.at[index, 'lyric'] = lyric
        dataset.at[index, 'mood'] = target_dict[dataset.at[index, 'mood']]

    dataset = dataset[['mood', 'lyric']]

    labels = dataset['mood'].tolist()
    lyrics = dataset['lyric'].tolist()

    return labels, lyrics

def load_dataset(dataset_path, duplicated_path):
    en_dataset = load_en_dataset(dataset_path, duplicated_path)

    remove_newline = True
    labels, lyrics = preprocess(en_dataset, remove_newline) 

    return np.array(labels), np.array(lyrics)

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

def to_DataLoader(input_ids, attention_masks, labels, hyperparemeters):
    dataset = TensorDataset(input_ids, attention_masks, labels)

    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), 
                            batch_size=hyperparemeters['model']['batch_size'])
    
    return dataloader

class XLNetForMultiLabelSequenceClassification(torch.nn.Module):
    
    def __init__(self, num_labels=4):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        #last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        #pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        output = self.classifier(mean_last_hidden_state)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(output, labels.float())
            return loss
        else:
            probs = torch.softmax(output, dim=1)
            return probs
        
    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True
    
    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector 
        """
        last_hidden_state = last_hidden_state[0]
        # TODO it can be done in different ways, not only mean
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_model(model, output_dir, epochs, lowest_eval_loss, train_loss_set, valid_loss_set):
    
    now = datetime.datetime.now()
    file_name = now.strftime("%Y-%m-%d_%H-%M-%S") + f'_after_{epochs}_epoch' + '.pt'
    output_path = os.path.join(output_dir, file_name)

    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs': epochs,
        'lowest_eval_loss': lowest_eval_loss,
        'train_loss_set': train_loss_set,
        'valid_loss_set': valid_loss_set
    }, output_path)

    print(f'Model saved to {output_path}')


def load_model(path_to_model):
    checkpoint = torch.load(path_to_model)
    model_state_dict = checkpoint['model_state_dict']
    model = XLNetForMultiLabelSequenceClassification(num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)

    epochs = checkpoint['epochs']
    lowest_eval_loss = checkpoint['lowest_eval_loss']
    train_loss_set = checkpoint['train_loss_set']
    valid_loss_set = checkpoint['valid_loss_set']

    return model, epochs, lowest_eval_loss, train_loss_set, valid_loss_set

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

        # ========================================
        #               Training
        # ========================================

        print("")
        print(f"Epoch {actual_epoch}/{hyperparameters['model']['epochs']}")
        print('Training...')

        t0 = time.time()

        model.train()

        tr_loss = 0
        num_train_samples = 0

        y_pred_train = []
        y_true_train = []

        for step, batch in enumerate(train_dataloader):
            if step % 3 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.')

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            tr_loss += loss.item()
            num_train_samples += b_labels.size(0)

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            #checking accuracy on training data
            output = model(b_input_ids, attention_mask=b_input_mask)
            output = output.detach().cpu().numpy()

            for predict in np.argmax(output, axis=1):
                y_pred_train.append(predict)

            labels_ids = b_labels.to('cpu').numpy()    
            for label in np.argmax(labels_ids, axis=1):
                y_true_train.append(label)


        epoch_train_loss = tr_loss/num_train_samples
        train_loss_set.append(epoch_train_loss)

        training_time = format_time(time.time() - t0)

        print("")
        print(f"  Average training loss: {round(epoch_train_loss, 3)}")
        print(f"  Training epcoh took: {training_time}")

        train_acc = accuracy_score(y_true_train, y_pred_train)

        print("")
        print(f"  Training accuracy: {round(train_acc, 3)}")

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")
        
        t0 = time.time()

        model.eval()

        eval_loss = 0
        num_eval_samples = 0

        y_pred_valid = []
        y_true_valid = []

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                loss = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

                eval_loss += loss.item()
                num_eval_samples += b_labels.size(0)

                #checking accuracy on validation data
                output = model(b_input_ids, attention_mask=b_input_mask)
                output = output.detach().cpu().numpy()

                for predict in np.argmax(output, axis=1):
                    y_pred_valid.append(predict)

                labels_ids = b_labels.to('cpu').numpy()    
                for label in np.argmax(labels_ids, axis=1):
                    y_true_valid.append(label)


        
        epoch_eval_loss = eval_loss/num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

        validation_time = format_time(time.time() - t0)

        print(f"  Validation Loss: {epoch_eval_loss}")
        print(f"  Validation took: {validation_time}")

        valid_acc = accuracy_score(y_true_valid, y_pred_valid)

        print("")
        print(f"  Validation accuracy: {valid_acc}")

        training_stats.append(
            {
                'epoch': actual_epoch,
                'Training Loss': epoch_train_loss,
                'Valid. Loss': epoch_eval_loss,
                'Training Accuracy': train_acc,
                'Validation Accuracy': valid_acc,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        # if lowest_eval_loss == None:
        #     lowest_eval_loss = epoch_eval_loss
        #     print(f'Best performance achived at epoch {actual_epoch} with validation loss of {lowest_eval_loss}')
        #     # save_model(model, hyperparameters['model_save_path'],
        #             #    actual_epoch, lowest_eval_loss, train_loss_set, valid_loss_set)
        # else:
        #     if epoch_eval_loss < lowest_eval_loss:
        #         lowest_eval_loss = epoch_eval_loss
        #         print(f'At epoch {actual_epoch} better performance was achived with validation loss of {lowest_eval_loss}')
                # save_model(model, hyperparameters['model_save_path'],
                        #    actual_epoch, lowest_eval_loss, train_loss_set, valid_loss_set)
    
    print("")
    print("Training complete!")
    print(f"Total training took {format_time(time.time()-total_t0)}")

    return model, train_loss_set, valid_loss_set, training_stats

def test_model(model,test_dataloader):
    
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    y_pred = []
    y_true = []

    print("")
    print("Running Testing...")

    for batch in test_dataloader:
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

    print("Testing complete!")
    print("")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=3))

def simple_run(hyperparemeters):

    dataset_path = os.path.join('..', 'database', 'lyrics')
    duplicated_path = os.path.join('database', 'removed_rows.json') 

    labels, lyrics = load_dataset(dataset_path, duplicated_path)

    tokienizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case = hyperparameters['tokenizer']['do_lower_case'])

    input_ids = tokenize_inputs(hyperparemeters, lyrics, tokienizer)
    attention_masks = create_attention_masks(input_ids)
    
    train_input_ids, test_input_ids, train_attention_masks, test_attention_masks, train_labels, test_labels = train_test_split(input_ids, 
                                                                                                                               attention_masks,
                                                                                                                               labels, 
                                                                                                                               random_state=SEED, test_size=0.3)


    test_input_ids, val_input_ids, test_attention_masks, val_attention_masks, test_labels, val_labels = train_test_split(test_input_ids, 
                                                                                                                         test_attention_masks,
                                                                                                                         test_labels, 
                                                                                                                         random_state=SEED, test_size=0.5)

    train_input_ids, train_attention_masks, train_labels = to_tensor(train_input_ids, train_attention_masks, train_labels)
    val_input_ids, val_attention_masks, val_labels = to_tensor(val_input_ids, val_attention_masks, val_labels)
    test_input_ids, test_attention_masks, test_labels = to_tensor(test_input_ids, test_attention_masks, test_labels)

    train_dataloader = to_DataLoader(train_input_ids, train_attention_masks, train_labels, hyperparemeters)
    val_dataloader = to_DataLoader(val_input_ids, val_attention_masks, val_labels, hyperparemeters)
    test_dataloader = to_DataLoader(test_input_ids, test_attention_masks, test_labels, hyperparemeters)

    model = XLNetForMultiLabelSequenceClassification(num_labels=hyperparemeters['model']['num_labels'])

    optimizer = AdamW(model.parameters(), 
                      lr=hyperparemeters['model']['lr'], 
                      weight_decay=hyperparameters['model']['weight_decay'],
                      correct_bias=hyperparameters['model']['correct_bias'], 
                      )
    
    model, train_loss_set, valid_loss_set, training_stats = train(model, optimizer, train_dataloader, val_dataloader, hyperparemeters)

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
                                'do_lower_case': True,
                                'num_embeddings': 256,
                            },
                            'model':{
                                'num_labels': 4,
                                'batch_size': 1024,
                                'lr': 5e-6,
                                'weight_decay': 0.01,
                                'correct_bias': False,
                                'epochs': 1,
                            },
                            'model_save_path': model_save_folder
                        }
        
        simple_run(hyperparameters)
    
    elif args.test_model:

        hyperparameters = {
                            'tokenizer':{
                                'do_lower_case': True,
                                'num_embeddings': 128,
                            },
                            'model':{
                                'num_labels': 4,
                                'batch_size': 32,
                            }
                            }
        
        path_to_model = os.path.join('models', 'lyric', 'xlnet', '2023-06-07_20-05-40.pt')

        model, _, _, _, _ = load_model(path_to_model)
        

        dataset_path = os.path.join('..', 'database', 'lyrics')
        duplicated_path = os.path.join('database', 'removed_rows.json') 

        labels, lyrics = load_dataset(dataset_path, duplicated_path)

        tokienizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case = hyperparameters['tokenizer']['do_lower_case'])

        input_ids = tokenize_inputs(hyperparameters, lyrics, tokienizer)
        attention_masks = create_attention_masks(input_ids)


        _, test_input_ids, _, test_attention_masks, _, test_labels = train_test_split(input_ids, 
                                                                                        attention_masks,
                                                                                        labels, 
                                                                                        random_state=SEED, test_size=0.3)
    

        test_input_ids, _, test_attention_masks, _, test_labels, _ = train_test_split(test_input_ids, 
                                                                                        test_attention_masks,
                                                                                        test_labels, 
                                                                                        random_state=SEED, test_size=0.5)
        
        test_input_ids, test_attention_masks, test_labels = to_tensor(test_input_ids, test_attention_masks, test_labels)

        test_dataloader = to_DataLoader(test_input_ids, test_attention_masks, test_labels, hyperparameters)

        test_model(model, test_dataloader, hyperparameters)    
    
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
                                            'epochs': 4,
                                        },
                                    }

                    simple_run(hyperparameters)

                    for key, value in hyperparameters.items():
                        print(f"{key} : {value}")

                    print('***************************************************\n\n')
                    
           
    