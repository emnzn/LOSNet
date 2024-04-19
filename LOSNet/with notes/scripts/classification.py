import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from multimodal import MultimodalClassifierDataset, LOSNetWeighted, collation
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel
from sklearn.preprocessing import OneHotEncoder
from joblib import dump
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, balanced_accuracy_score

base_path = 'data/split/with-outliers/combined'

static_train = pd.read_csv(f'{base_path}/static_train.csv')
static_val = pd.read_csv(f'{base_path}/static_val.csv')

dynamic = pd.read_csv('data/dynamic_cleaned.csv')
dynamic_train = dynamic[dynamic['id'].isin(static_train['id'])].copy()
dynamic_val = dynamic[dynamic['id'].isin(static_val['id'])].copy()


def truncate_and_average(df, id_col, max_records=4):
    df_sorted = df.sort_values(by=[id_col, 'charttime'])

    def process_group(group):
        if len(group) > max_records:
            average_data = group.iloc[:-max_records].drop(columns=['charttime']).mean().to_dict()
            average_data[id_col] = group[id_col].iloc[0]
            average_row = pd.DataFrame([average_data])

            return pd.concat([average_row, group.tail(max_records)], ignore_index=True)
        else:
            return group

    return df_sorted.groupby(id_col).apply(process_group).reset_index(drop=True)


dynamic_train = truncate_and_average(dynamic_train, 'id')
dynamic_val = truncate_and_average(dynamic_val, 'id')

dynamic_features = ['aniongap', 'bicarbonate', 'bun', 'calcium', 'chloride',
                    'creatinine', 'glucose', 'sodium', 'potassium']

scaler = StandardScaler()

dynamic_train[dynamic_features] = scaler.fit_transform(dynamic_train[dynamic_features])
dynamic_val[dynamic_features] = scaler.transform(dynamic_val[dynamic_features])

dump(scaler, './scalers/dynamic_scaler.joblib')

id_lengths_train = dynamic_train['id'].value_counts().to_dict()
dynamic_train = dynamic_train.sort_values(by=['id', 'charttime'])
dynamic_train = dynamic_train.apply(lambda x: list(x[dynamic_features]), axis=1).groupby(dynamic_train['id']).agg(list)

id_lengths_val = dynamic_val['id'].value_counts().to_dict()
dynamic_val = dynamic_val.sort_values(by=['id', 'charttime'])
dynamic_val = dynamic_val.apply(lambda x: list(x[dynamic_features]), axis=1).groupby(dynamic_val['id']).agg(list)

notes = pd.read_csv('data/notes_cleaned.csv')
notes = notes[['id', 'charttime', 'text', 'interval']]

notes_train = notes[notes['id'].isin(static_train['id'])].copy()
notes_val = notes[notes['id'].isin(static_val['id'])].copy()

train_data = MultimodalClassifierDataset(
    static=static_train, dynamic=dynamic_train,
    id_lengths=id_lengths_train, notes=notes_train
    )
validation_data = MultimodalClassifierDataset(
    static=static_val, dynamic=dynamic_val,
    id_lengths=id_lengths_val, notes=notes_val
    )

seed_value = 24
num_lstm_cells = 1
out_features = 3

torch.manual_seed(seed_value)

cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

train_loader = DataLoader(train_data, batch_size=200, shuffle=True, collate_fn=collation)
val_loader = DataLoader(validation_data, batch_size=400, shuffle=True, collate_fn=collation)

text_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

freeze_status = False
for name, param in text_model.named_parameters():
    if name.startswith("encoder.layer.10"):
        freeze_status = True
    param.requires_grad = freeze_status

static_input_size = 14
dynamic_input_size = 9
hidden_size = 64
print(f'total fc input size: {static_input_size + hidden_size}')

model = LOSNetWeighted(dynamic_input_size=dynamic_input_size, static_input_size=static_input_size,
                       out_features=out_features, hidden_size=hidden_size,
                       text_model=text_model, task='cls')

for name, param in model.text_model.named_parameters():
    print(f"Layer: {name}, Frozen: {not param.requires_grad}")

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = model.to(device)
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model.to(device),
#                             device_ids=range(torch.cuda.device_count()))  # Wrap the model in DataParallel

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

epochs = 200
f1_type = 'weighted'
training_loss = []
validation_loss = []
train_f1_scores = []
val_f1_scores = []
train_acc_scores = []
val_acc_scores = []
patience = 10
stagnation = 0


def gen_dir(num_lstm_cells, dataset_type='combined_regression', hidden_size=32, trial_num=1):
    loss_base_path = f'./losses/trial_num_{trial_num}/{dataset_type}/{num_lstm_cells}_cells_{epochs}_epochs_{hidden_size}_hidden_size'
    model_save_path = f'./saved-models/trial_num_{trial_num}/{num_lstm_cells}_cells_{dataset_type}_{epochs}_epochs_{hidden_size}_hidden_size'
    tensorboard_path = f'./tensorboard/runs/trial_num_{trial_num}/{dataset_type}_static_dynamic_{num_lstm_cells}_cells_{epochs}_epochs_{hidden_size}_hidden_size'

    if not os.path.exists(loss_base_path):
        os.makedirs(loss_base_path)
        print(f"Created loss directory: {loss_base_path}")
    else:
        print(f"Directory for loss exists: {loss_base_path}")

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        print(f"Created model directory: {model_save_path}")
    else:
        print(f"Directory for model exists: {model_save_path}")

    return loss_base_path, model_save_path, tensorboard_path


loss_base_path, model_save_path, tensorboard_path = gen_dir(num_lstm_cells,
                                                            'combined_classification_expansion_fc_truncated',
                                                            hidden_size=hidden_size, trial_num=0)

writer = SummaryWriter(tensorboard_path)

for epoch in range(1, epochs + 1):

    print(f'training epoch: [{epoch}/{epochs}]')
    model.train()
    training_loss_epoch = 0
    all_true_labels = []
    all_predicted_labels = []

    for step, batch in enumerate(train_loader):
        packed_dynamic_X, notes_X, notes_intervals, static_batch, los = batch

        packed_dynamic_X = packed_dynamic_X.to(device)
        los = los.to(device)
        static_X_gpu = static_batch.to(device)

        notes_X_gpu = []
        for notes in notes_X:
            notes_gpu = {key: value.to(device) for key, value in notes.items()}
            notes_X_gpu.append(notes_gpu)

        outputs = model(packed_dynamic_X, notes_X_gpu, notes_intervals, static_X_gpu)
        predicted_labels = torch.argmax(outputs, dim=1).float()
        true_labels = torch.argmax(los, dim=1).float()

        loss = criterion(outputs, true_labels)
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + step)
        training_loss_epoch += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_true_labels.extend(true_labels.cpu().numpy())
        all_predicted_labels.extend(predicted_labels.cpu().numpy())

        if step % max(1, round(len(train_loader) * 0.1)) == 0:
            print(f'step: [{step + 1}/{len(train_loader)}] | loss: {loss.item():.4}')

            if step + 1 == 1 and epoch == 1:
                with open(f'{loss_base_path}/loss_step.txt', 'w') as loss_step_f:
                    loss_step_f.write(f'{loss.item():.4f}\n')

            else:
                with open(f'{loss_base_path}/loss_step.txt', 'a') as loss_step_f:
                    loss_step_f.write(f'{loss.item():.4f}\n')

    avg_training_loss_epoch = training_loss_epoch / len(train_loader)
    writer.add_scalar('Loss/train_avg', avg_training_loss_epoch, epoch)

    training_loss.append(avg_training_loss_epoch)
    print(f'\nTraining epoch loss: {avg_training_loss_epoch:.4f}')

    train_f1_score = f1_score(all_true_labels, all_predicted_labels, average=f1_type)
    train_f1_scores.append(round(train_f1_score, 4))
    print(f'Training {f1_type} F1 epoch score: {train_f1_score:.4f}')
    writer.add_scalar('F1/train', train_f1_score, epoch)

    train_acc_score = balanced_accuracy_score(all_true_labels, all_predicted_labels)
    train_acc_scores.append(round(train_acc_score, 4))
    print(f'Training balanced accuracy epoch score: {train_acc_score:.4f}\n')
    writer.add_scalar('Accuracy/train', train_acc_score, epoch)

    if epoch == 1:
        with open(f'{loss_base_path}/training_loss_epoch.txt', 'w') as loss_epoch_train_f:
            loss_epoch_train_f.write(f'{avg_training_loss_epoch:.4f}\n')

    else:
        with open(f'{loss_base_path}/training_loss_epoch.txt', 'a') as loss_epoch_train_f:
            loss_epoch_train_f.write(f'{avg_training_loss_epoch:.4f}\n')

    print(f'validation epoch: [{epoch}/{epochs}]')
    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        validation_loss_epoch = 0
        val_all_true_labels = []
        val_all_predicted_labels = []

        for val_step, val_batch in enumerate(val_loader):
            packed_dynamic_X_val, notes_X_val, notes_intervals_val, static_val_batch, los_val = val_batch

            packed_dynamic_X_val = packed_dynamic_X_val.to(device)
            los_val = los_val.to(device)

            notes_X_val_gpu = []
            for notes_val in notes_X_val:
                notes_val_gpu = {key: value.to(device) for key, value in notes_val.items()}
                notes_X_val_gpu.append(notes_val_gpu)

            static_X_val_gpu = static_val_batch.to(device)

            val_outputs = model(packed_dynamic_X_val, notes_X_val_gpu, notes_intervals_val, static_X_val_gpu)

            val_predicted_labels = torch.argmax(val_outputs, dim=1).float()
            val_true_labels = torch.argmax(los_val, dim=1).float()

            val_loss = criterion(val_outputs, val_true_labels)
            writer.add_scalar('Loss/val', val_loss.item(), epoch * len(val_loader) + val_step)
            validation_loss_epoch += val_loss.item()

            val_all_true_labels.extend(val_true_labels.cpu().numpy())
            val_all_predicted_labels.extend(val_predicted_labels.cpu().numpy())

        avg_validation_loss = validation_loss_epoch / len(val_loader)
        writer.add_scalar('Loss/val_avg', avg_validation_loss, epoch)
        print(f'Validation epoch loss: {avg_validation_loss:.4f}')

        val_f1_score = f1_score(val_all_true_labels, val_all_predicted_labels, average=f1_type)
        print(f'Validation {f1_type} F1 epoch score: {val_f1_score:.4f}')
        writer.add_scalar('F1/val', val_f1_score, epoch)

        val_acc_score = balanced_accuracy_score(val_all_true_labels, val_all_predicted_labels)
        print(f'Validation balanced accuracy epoch score: {val_acc_score:.4f}\n')
        writer.add_scalar('Accuracy/val', val_acc_score, epoch)

        if len(validation_loss) == 0 or (avg_validation_loss < min(validation_loss)):
            stagnation = 0
            torch.save(model.state_dict(), f'{model_save_path}/lowest_loss_model.pth')
            print(f'new minimum validation loss')
            print(f'model saved\n')

        if len(val_f1_scores) == 0 or (val_f1_score > max(val_f1_scores)):
            torch.save(model.state_dict(), f'{model_save_path}/highest_f1_model.pth')
            print(f'new max {f1_type} F1 score')
            print(f'model saved\n')

        if len(val_acc_scores) == 0 or (val_acc_score > max(val_acc_scores)):
            torch.save(model.state_dict(), f'{model_save_path}/highest_accuracy_model.pth')
            print(f'new max balanced accuracy score')
            print(f'model saved\n')

        else:
            stagnation += 1

        validation_loss.append(avg_validation_loss)
        val_f1_scores.append(round(val_f1_score, 4))
        val_acc_scores.append(round(val_acc_score, 4))

        if epoch == 1:
            with open(f'{loss_base_path}/validation_loss_epoch.txt', 'w') as loss_epoch_val_f:
                loss_epoch_val_f.write(f'{avg_validation_loss:.4f}\n')

        else:
            with open(f'{loss_base_path}/validation_loss_epoch.txt', 'a') as loss_epoch_val_f:
                loss_epoch_val_f.write(f'{avg_validation_loss:.4f}\n')

        if stagnation >= patience:
            print(f'No improvement over {patience} epochs')
            print('Early stopping\n')
            break

    model.train()

    print('===============================\n')

writer.close()
print(f'min training loss: {min(training_loss):.4f}')
print(f'min validation loss: {min(validation_loss):.4f}\n')

print(f'max training f1: {max(train_f1_scores):.4f}')
print(f'max validation f1: {max(val_f1_scores):.4f}\n')

print(f'max training acc: {max(train_acc_scores):.4f}')
print(f'max validation acc: {max(val_acc_scores):.4f}')

