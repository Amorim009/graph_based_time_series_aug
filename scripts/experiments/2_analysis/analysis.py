import pandas as pd
import plotnine as p9

from utils.analysis import to_latex_tab, read_results, THEME

df = read_results('mase')
df = df.drop(columns=['derived_ensemble', 'derived'])

COLUMN_MAP = {
    'MagnitudeWarping': 'M-Warp',
    'TimeWarping': 'T-Warp',
    'SeasonalMBB': 'MBB',
    'Jittering': 'Jitter',
    'original': 'Original',
    'SeasonalNaive': 'SNaive',
    'derived': 'QGTS(D)',
    'QGTSE': 'Grasynda(E)',
    'QGTS': 'Grasynda',
}

APPROACH_COLORS = [
    '#2c3e50',  # Dark slate blue
    '#34558b',  # Royal blue
    '#4b7be5',  # Bright blue
    '#6db1bf',  # Light teal
    '#bf9b7a',  # Warm tan
    '#d17f5e',  # Warm coral
    '#c44536',  # Burnt orange red
    '#8b1e3f',  # Deep burgundy
    '#472d54',  # Deep purple
    '#855988',  # Muted mauve
    '#2d5447',  # Forest green
    '#507e6d'  # Sage green
]

df = df.rename(columns=COLUMN_MAP)
df = df[['Original', 'Grasynda', 'Grasynda(E)', 'DBA',
         'Jitter', 'M-Warp', 'MBB', 'Scaling', 'T-Warp', 'TSMixup',
         'SNaive', 'ds', 'model']]

# overall details on table
perf_by_all = df.groupby(['model', 'ds']).mean(numeric_only=True)

avg_perf = perf_by_all.reset_index().groupby('model').mean(numeric_only=True)
avg_rank = perf_by_all.rank(axis=1).reset_index(level='model').groupby('model').mean(numeric_only=True).round(2)

og = perf_by_all['Original']
effectiveness = perf_by_all.apply(lambda x: (x < og).astype(int), axis=0).mean()

# perf_by_mod = df.groupby(['model']).mean(numeric_only=True)
# avg_score = perf_by_mod.mean().values

# avg_rank = perf_by_all.rank(axis=1).mean().round(2).values

# perf_by_mod.loc[('All', 'Average'), :] = avg_score
# perf_by_all.loc[('All', 'Avg. Rank'), :] = avg_rank
perf_by_all.loc[('All', 'Effectiveness'), :] = effectiveness.round(2)

perf_by_all.index = pd.MultiIndex.from_tuples(
    [(f'\\rotatebox{{90}}{{{x[0]}}}', x[1]) for x in perf_by_all.index]
)

tex_tab = to_latex_tab(perf_by_all, 4, rotate_cols=True)
print(tex_tab)

# grouped bar plot
# ord = avg_rank.mean().sort_values().index.tolist()
ord = avg_perf.mean().sort_values().index.tolist()
# scores_df = avg_rank.reset_index().melt('model')
scores_df = avg_perf.reset_index().melt('model')
scores_df.columns = ['Model', 'Method', 'Average Rank']
scores_df['Method'] = pd.Categorical(scores_df['Method'], categories=ord)

plot = \
    p9.ggplot(data=scores_df,
              mapping=p9.aes(x='Model',
                             y='Average Rank',
                             fill='Method')) + \
    p9.geom_bar(position='dodge',
                stat='identity',
                width=0.9) + \
    THEME + \
    p9.theme(axis_title_y=p9.element_text(size=12),
             axis_title_x=p9.element_blank(),
             axis_text=p9.element_text(size=12)) + \
    p9.scale_fill_manual(values=APPROACH_COLORS)

plot.save('assets/results/outputs/mase_by_model_op2.pdf', height=5, width=12)
