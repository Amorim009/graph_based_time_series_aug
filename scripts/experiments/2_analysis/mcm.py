import sys

sys.path.append(r'C:\Users\lhenr\Desktop\graph_based_time_series_aug')

import numpy as np
import pandas as pd



from utils.load_data.config import DATASETS
from utils.load_data.config import DATA_GROUPS

from src.workflow import ExpWorkflow
from utils.load_data.base import LoadDataset
from src.qgraph_ts import QuantileGraphTimeSeriesGenerator as QGTSGen
from src.qgraph_ts import QuantileDerivedTimeSeriesGenerator as DerivedGen
from pytorch_lightning import Trainer
from multi_comp_matrix import MCM
from pytorch_lightning import Trainer



paths = [
    "assets/results/M3_Monthly_KAN.csv",
    "assets/results/Tourism_Monthly_KAN.csv",
    "assets/results/Gluonts_m1_quarterly_KAN.csv",
    "assets/results/M3_Quarterly_KAN.csv",
    "assets/results/Tourism_Quarterly_KAN.csv",
    "assets/results/Gluonts_m1_monthly_KAN.csv"
]

dfs = [pd.read_csv(path) for path in paths]
for i, df in enumerate(dfs):
    df['Dataset'] = paths[i].split('/')[-1].replace('_KAN.csv', '')

global_df = pd.concat(dfs, ignore_index=True)
selected_columns = ["qgts(25)", "qgtse(50)", "SeasonalNaive", "original", "Scaling", "SeasonalMBB"]
filtered_df = global_df[selected_columns + ['Dataset']]
filtered_df.to_csv("assets/results/global_mase_results.csv", index=False)


#print(filtered_df)



df = pd.read_csv("assets/results/global_mase_results.csv")
df = df.drop(columns=['Dataset'])
test_df = df.iloc[:200, :]
print(df.shape)
print(test_df.shape)

MCM.compare(
    df_results= test_df,
    output_dir="assets/results/MCM",
    used_statistic="MASE",
    order_WinTieLoss="lower",
    order_better="increasing",
    #dataset_column="Dataset",
    pvalue_test="wilcoxon",
    pvalue_threshold=0.05,
    colormap="coolwarm_r",
    win_label="r<c",
    loss_label="r>c",
    tie_label="r=c",
    csv_savename="mase_comparison.csv",
    png_savename="mase_heatmap.png",
    include_ProbaWinTieLoss=True,
    bayesian_rope=0.05
)