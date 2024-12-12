import numpy as np
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import KAN, MLP
from utilsforecast.losses import mase, smape
from utilsforecast.evaluation import evaluate
from functools import partial

from utils.load_data.config import DATASETS
from utils.load_data.config import DATA_GROUPS

from extra.utils import DataUtils
from extra.method import QuantileGraphGenerator

data_name, group = DATA_GROUPS[4]
print(data_name, group)

# LOADING DATA AND SETUP
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, min_n_instances=min_samples)

print(df['unique_id'].value_counts())
print(df.shape)

# SPLITS AND MODELS
train, test = DataUtils.train_test_split(df, horizon)

# + datasets
# adicionar seasonalnaive
# + geradores de dados
# analise + profunda
# + modelos (KAN,MLP, NHITS, etc)


gen = QuantileGraphGenerator(n_quantiles=10,
                             quantile_on='remainder',
                             period=12,
                             ensemble_transitions=False)

egen = QuantileGraphGenerator(n_quantiles=10,
                              quantile_on='remainder',
                              period=12,
                              ensemble_transitions=True)

synth_df = gen.transform(train)
synth_df2 = egen.transform(train)

train_aug = pd.concat([train, synth_df]).reset_index(drop=True)
train_aug2 = pd.concat([train, synth_df2]).reset_index(drop=True)

# using original train
nf = NeuralForecast(models=[MLP(h=horizon,
                                input_size=n_lags,
                                max_steps=1000,
                                accelerator='cpu',
                                alias='original')], freq=freq_str)
nf.fit(df=train, val_size=horizon)
fcst_orig = nf.predict()

nf2 = NeuralForecast(models=[MLP(h=horizon,
                                 input_size=n_lags,
                                 max_steps=1000,
                                 accelerator='cpu', alias='aug')], freq=freq_str)
nf2.fit(df=train_aug, val_size=horizon)
fcst_ext = nf2.predict()

nf3 = NeuralForecast(models=[MLP(h=horizon, input_size=n_lags, max_steps=1000, accelerator='cpu', alias='aug2')], freq=freq_str)
nf3.fit(df=train_aug2, val_size=horizon)
fcst_ext2 = nf3.predict()

test = test.merge(fcst_orig.reset_index(), on=['unique_id', 'ds'], how="left")
test = test.merge(fcst_ext.reset_index(), on=['unique_id', 'ds'], how="left")
test = test.merge(fcst_ext2.reset_index(), on=['unique_id', 'ds'], how="left")
evaluation_df = evaluate(test, [partial(mase, seasonality=freq_int), smape], train_df=train)

print(evaluation_df.query('metric=="mase"').mean(numeric_only=True))
print(evaluation_df.query('metric=="smape"').mean(numeric_only=True))
#  OWA
print(evaluation_df.mean(numeric_only=True))

test['unique_id'].value_counts()
