import pandas as pd
from scipy.stats import wilcoxon

from utils.analysis import to_latex_tab, read_results, THEME

df = read_results('mase')
df = df.drop(columns=['derived_ensemble', 'derived'])


df_m1q = df.query('ds=="M3-Q"')
df_m1q_nhits = df_m1q.query('model=="NHITS"')

x = df_m1q['QGTSE']
y = df_m1q['original']

st = wilcoxon(x=x,y=y)
st.pvalue


