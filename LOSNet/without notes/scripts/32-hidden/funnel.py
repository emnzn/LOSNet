import torch
import numpy as np
import torch.nn as nn
from torch.nn import LSTM
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class MultimodalRegressorDataset(Dataset):
    '''
    Parameters
    ----------
    static: the static DataFrame
    dynamic: dynamic dataset where each timestep is concatenated into one dimension
        ```
            id
            20008098    [[21.0, 20.0, 21.0, 8.9, 100.0, 0.9, 178.0, 13...
            20013244    [[13.0, 29.0, 17.0, 8.9, 103.0, 1.1, 127.0, 14...
            20015730    [[11.0, 25.0, 17.0, 8.1, 112.0, 1.6, 121.0, 14...
            20020562    [[12.0, 22.0, 21.0, 8.2, 104.0, 2.2, 91.0, 134...
            20021110    [[16.0, 27.0, 32.0, 9.7, 103.0, 1.2, 88.0, 141...
        ```

    id_lengths: a dictionary where the key is the patient_id and the value is the true length the time series associated with each patient id (to be used for packed padding)
        ```
            {
                20008098: 9,
                20013244: 7,
                20015730: 10,
                20020562: 10,
                20021110: 10,
                20022095: 6,
                20022465: 6,
                20024177: 7
            }

    Outputs
    -------
    packed_dynamic_X: A sequence of time steps, dynamically packed and padded, representing data for a specific patient
    '''

    def __init__(self, static, dynamic, id_lengths, ohe_cols=None):
        self.static = static
        self.dynamic = dynamic
        self.id_lengths = id_lengths

        if ohe_cols is None:
            self.ohe_cols = [
                'los_icu_binned_1 to 2 days', 
                'los_icu_binned_2 to 4 days',
                'los_icu_binned_4+ days'
            ]
        else:
            self.ohe_cols = ohe_cols 

    def __len__(self):
        return len(self.static)
    
    def __getitem__(self, idx):
        patient_id = self.static.iloc[idx]['id']
        to_drop = self.ohe_cols + ['id']
        static_X = self.static.iloc[idx].drop(to_drop)
        static_y = self.static.iloc[idx]['los_icu']

        # time series
        dynamic_X = self.dynamic[patient_id]
        dynamic_X = torch.tensor(dynamic_X, dtype=torch.float32)
        patient_timesteps = self.id_lengths[patient_id]

        # los
        los = [static_y]
        return dynamic_X, patient_timesteps, static_X, los
    
class MultimodalClassifierDataset(MultimodalRegressorDataset):
    def __init__(self, static, dynamic, id_lengths, ohe_cols=None):
        super(MultimodalClassifierDataset, self).__init__(
            static, dynamic, id_lengths, ohe_cols
        )
            
        self.static_dict = {idx: row[self.ohe_cols].astype(np.float32).to_list() for idx, row in static.set_index('id').iterrows()}

    def __len__(self):
        return len(self.static)
    
    def __getitem__(self, idx):
        patient_id = self.static.iloc[idx]['id']
        to_drop = self.ohe_cols + ['id']

        static_X = self.static.iloc[idx].drop(to_drop).to_numpy()
        static_y = self.static.iloc[idx][self.ohe_cols].to_numpy()

        # time series
        dynamic_X = self.dynamic[patient_id]
        dynamic_X = torch.tensor(dynamic_X, dtype=torch.float32)
        patient_timesteps = self.id_lengths[patient_id]

        # los
        los = static_y
        return dynamic_X, patient_timesteps, static_X, los
    
def collation(batch):
    dynamic_X_batch, patient_timesteps, static_X_batch, los_batch = zip(*batch)

    padded_dynamic_batch = pad_sequence(dynamic_X_batch, batch_first=True, padding_value=0.0)
    packed_dynamic_X_batch = pack_padded_sequence(input=padded_dynamic_batch, lengths=patient_timesteps, batch_first=True, enforce_sorted=False)

    static_X_batch = np.array(static_X_batch)
    static_X_batch = torch.tensor(static_X_batch, dtype=torch.float32)

    los_batch = np.array(los_batch)
    los_batch = torch.tensor(los_batch, dtype=torch.float32)

    return packed_dynamic_X_batch, static_X_batch, los_batch

class LOSNet(nn.Module):
    '''
    time_series_model: expects an input of packed padded sequences
    '''
    def __init__(
            self, static_input_size, dynamic_input_size, out_features, 
            hidden_size, batch_first=True, 
            num_cells=1, **kwargs
            ):
    
        super(LOSNet, self).__init__(**kwargs)
        
        self.time_series_model = LSTM(input_size=dynamic_input_size, hidden_size=hidden_size, batch_first=batch_first, num_layers=num_cells)

        self.ht_layer_norm = nn.LayerNorm(normalized_shape=hidden_size)

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_size + static_input_size),
            nn.Linear(in_features=hidden_size + static_input_size, out_features=50, bias=True),
            nn.LayerNorm(50),
            nn.ReLU(),

            nn.Linear(in_features=50, out_features=20, bias=True),
            nn.LayerNorm(20),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(in_features=20, out_features=10, bias=True),
            nn.LayerNorm(10),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(in_features=10, out_features=out_features, bias=True),
        )

    def forward(self, packed_dynamic_X_batch, static_X_batch):
        _, (ht, _) = self.time_series_model(packed_dynamic_X_batch)
        ht = ht[-1]
        ht = self.ht_layer_norm(ht)

        sp = static_X_batch
        
        combined_representation = torch.cat((ht, sp), dim=1)
        
        logits = self.fc(combined_representation)
        y_pred = logits 

        return y_pred