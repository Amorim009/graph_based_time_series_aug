from utils.load_data.config import DATASETS
from utils.load_data.config import DATA_GROUPS
from utils.load_data.base import LoadDataset
from src.qgraph_ts import QuantileGraphTimeSeriesGenerator as QGTSGen

data_name, group = DATA_GROUPS[0]
print(data_name, group)
N_QUANTILES = 25

# LOADING DATA AND SETUP
data_loader = DATASETS[data_name]
min_samples = data_loader.min_samples[group]
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group,
                                                                      min_n_instances=min_samples)

print(df['unique_id'].value_counts())
print(df.shape)

# DATA SPLITS
train, _ = LoadDataset.train_test_split(df, horizon)

# QGTS
qgts_gen = QGTSGen(n_quantiles=N_QUANTILES,
                   quantile_on='remainder',
                   period=freq_int,
                   ensemble_transitions=False)

qgts_df = qgts_gen.transform(train)

qgts_gen.matrix_to_edgelist('ID90').to_csv('assets/results/plot_data/example.csv', index=False)
