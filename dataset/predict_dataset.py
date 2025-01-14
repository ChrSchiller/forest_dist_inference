from torch.utils.data import Dataset
import torch
import numpy as np

class PredictDataset(Dataset):
    """Time Series Disturbance dataset."""

    def __init__(self, locations, input_data, input_dates, seq_len, feature_num):
        """
        Args:
            labels_ohe (string): dataframe with integer class labels
            root_dir (string): Directory with all the time series files.
        """

        self.locations = locations
        self.input_data = input_data
        self.input_dates = input_dates
        self.seq_len = seq_len
        self.dimension = feature_num

    def __len__(self):
        return self.locations.shape[0]  # number of samples in the dataset

    def __getitem__(self, idx):
        ts = self.input_data[self.locations[idx], :, :]
        ### should not be masked values, but 'real' values
        ### dimensions of ts: [seq_len, feature_num]

        ### remove complete observations if any band value is nan
        ### (sometimes SWIR2 == np.nan although all other bands contain valid values)
        valid_obs_arr = ~np.isnan(ts).any(axis=1) # True means valid values, False means masked values
        input_dates = [self.input_dates[i] for i in range(len(self.input_dates)) if valid_obs_arr.tolist()[i]]
        ts = ts[valid_obs_arr, :]

        # get number of observations for further processing
        ts_length = ts.shape[0]

        ### we always take the LATEST seq_len observations,
        ### (in the original paper/code, it has been the FIRST seq_len observations)
        ### since they are considered most important
        bert_mask = np.zeros((self.seq_len,), dtype=int)
        bert_mask[-ts_length:] = 1

        ### day of year
        doy = np.zeros((self.seq_len,), dtype=int)

        ### here, we need to retrieve and assign the correct DOY values
        # get day of the year of startDate
        t0 = input_dates[0].timetuple().tm_yday
        input_dates = np.array([(date - input_dates[0]).days for date in input_dates]) + t0

        # BOA reflectances
        ts_origin = np.zeros((self.seq_len, self.dimension))
        ts_origin[-ts_length:, :] = ts[-ts_length:, :] / 10000.0  # division by 10k likely unnecessary because of z-transform
        doy[-ts_length:] = input_dates

        ### apply z-transformation on each band individually
        ### note that we can leave the DOY values as they are,
        ### since they are being transformed later in PositionalEncoding
        ts_origin[-ts_length:, :] = (ts_origin[-ts_length:, :] - ts_origin[-ts_length:, :].mean(axis=0)) / (ts_origin[-ts_length:, :].std(axis=0) + 1e-6)

        output = {"bert_input": ts_origin,
                  "bert_mask": bert_mask,
                  "time": doy
                  }

        return {key: torch.from_numpy(value) for key, value in output.items()}