import sys

sys.path.append(r'C:\Users\lhenr\Desktop\graph_based_time_series_aug')

import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from statsforecast.models import SeasonalNaive
from statsforecast import StatsForecast
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from functools import partial

from utils.load_data.config import DATASETS
from utils.load_data.config import DATA_GROUPS
from utils.config import SYNTH_METHODS, MODEL_CONFIG, MODELS
from src.workflow import ExpWorkflow
from utils.load_data.base import LoadDataset

from pytorch_lightning import Trainer

trainer = Trainer(accelerator='cpu')

data_name, group = DATA_GROUPS[1]

# LOADING DATA AND SETUP
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group,
                                                                      min_n_instances=min_samples)


num_series = df['unique_id'].nunique()
print(f"Number of time series: {num_series}")

avg_obs_per_series = df.groupby('unique_id').size().mean()
print(f"Average observations per time series: {avg_obs_per_series:.2f}")


n_quantiles = 4


df['quantile'] = df.groupby('unique_id')['y'].transform(
    lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
)

first_id = df['unique_id'].iloc[500]


quantile_counts = df[df['unique_id'] == first_id]['quantile'].value_counts().sort_index()

print(f"Quantile counts for unique_id = {first_id}:\n")
print(quantile_counts)