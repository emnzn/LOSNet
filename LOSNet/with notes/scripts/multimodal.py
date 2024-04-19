import torch
import torch.nn as nn
from torch.nn import LSTM
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F


class MultimodalClassifierDataset(Dataset):
    def __init__(self, static, dynamic, id_lengths, notes, los_bins=None):
        self.static = static
        self.dynamic = dynamic
        self.id_lengths = id_lengths
        self.notes = notes
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        if los_bins is None:
            self.los_bins = [
                'los_icu_binned_1 to 2 days',
                'los_icu_binned_2 to 4 days',
                'los_icu_binned_4+ days'
            ]
        else:
            self.los_bins = los_bins

        self.static_dict = {idx: row[self.los_bins].astype(np.float32).to_list() for idx, row in
                            static.set_index('id').iterrows()}

    def __len__(self):
        return len(self.static)

    def __getitem__(self, idx):
        patient_id = self.static.iloc[idx]['id']

        to_drop = self.los_bins + ['id']

        # time series
        patient_dynamic = self.dynamic[patient_id]
        patient_dynamic = torch.tensor(patient_dynamic, dtype=torch.float32)
        patient_timesteps = self.id_lengths[patient_id]

        # notes
        notes = self.notes[self.notes['id'] == patient_id]['text'].tolist()
        notes_intervals = self.notes[self.notes['id'] == patient_id]['interval'].to_numpy()
        notes_intervals = torch.tensor(notes_intervals, dtype=torch.float32)
        patient_notes = self.tokenizer(notes, return_tensors='pt', truncation=True,
                                       max_length=512, padding='max_length')

        # static
        static_X = self.static.iloc[idx].drop(to_drop).to_numpy()
        static_y = self.static.iloc[idx][self.los_bins].to_numpy()

        # los
        los = static_y

        return patient_dynamic, patient_timesteps, patient_notes, notes_intervals, static_X, los


class LOSNetWeighted(nn.Module):
    '''
    time_series_model: expects an input of packed padded sequences
    text_model: expects an input of dict with keys {'input_ids', 'token_type_ids', 'attention_mask'}
                of tokenized sequences
    '''

    def __init__(
            self, dynamic_input_size, static_input_size, out_features,
            hidden_size, text_model=None,
            decay_factor=0.1, batch_first=True,
            task='reg', **kwargs
    ):
        assert (task == 'reg' or task == 'cls'), 'task must be either `reg` or `cls`'

        super(LOSNetWeighted, self).__init__(**kwargs)
        self.decay_factor = decay_factor
        self.task = task

        self.time_series_model = LSTM(input_size=dynamic_input_size, hidden_size=hidden_size,
                                      batch_first=batch_first)
        self.ht_layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

        self.text_model = text_model if text_model is not None \
            else AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        self.text_model = torch.nn.DataParallel(self.text_model, device_ids=[0,1,2,3])

        self.fc1 = nn.Sequential(

            nn.Linear(in_features=hidden_size + 768 + static_input_size,
                      out_features=256,
                      bias=True),
            nn.LayerNorm(normalized_shape=256),
            nn.ReLU(),


            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.fc2 = nn.Linear(in_features=64,
                             out_features=out_features,
                             bias=True)

    def weighted_sum(self, embeddings, interval, decay_factor):
        device = embeddings.device
        weights = ((1 - decay_factor) ** interval).to(device)
        weighted_sum = torch.matmul(weights, embeddings)

        return weighted_sum

    def forward(self, packed_dynamic_X_batch, notes_X_batch, notes_intervals_batch, static_batch):
        _, (ht, _) = self.time_series_model(packed_dynamic_X_batch)
        ht = ht[-1]
        ht = self.ht_layer_norm(ht).to(self.device)

        embeddings = []
        for (patient_notes, notes_interval) in zip(notes_X_batch, notes_intervals_batch):
            patient_embeddings = self.text_model(**patient_notes).pooler_output
            weighted_sum = self.weighted_sum(embeddings=patient_embeddings, interval=notes_interval,
                                             decay_factor=self.decay_factor)
            embeddings.append(weighted_sum)

        zt = torch.stack(embeddings)
        zt = zt.to(self.device)

        st = static_batch
        st = st.to(self.device)

        combined_representation = torch.cat((ht, zt, st), dim=1)

        fc1_out = self.fc1(combined_representation)
        logits = self.fc2(fc1_out)
        y_pred = logits if self.task == 'reg' else F.softmax(logits, dim=-1)

        return y_pred


def collation(batch):
    dynamic_X_batch, patient_timesteps, \
        notes_X_batch, notes_intervals_batch, \
        static_batch, los_batch = zip(*batch)

    padded_dynamic_batch = pad_sequence(dynamic_X_batch, batch_first=True, padding_value=0.0)
    packed_dynamic_X_batch = pack_padded_sequence(input=padded_dynamic_batch, lengths=patient_timesteps,
                                                  batch_first=True, enforce_sorted=False)

    static_X_batch = np.array(static_batch)
    static_X_batch = torch.tensor(static_X_batch, dtype=torch.float32)

    los_batch = np.array(los_batch)
    los_batch = torch.tensor(los_batch, dtype=torch.float32)

    return packed_dynamic_X_batch, notes_X_batch, notes_intervals_batch, static_X_batch, los_batch

