import copy
from typing import Dict
from itertools import combinations

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL

from metaforecast.synth.generators.base import SemiSyntheticGenerator


class QuantileGraphTimeSeriesGenerator(SemiSyntheticGenerator):

    def __init__(self,
                 n_quantiles: int,
                 quantile_on: str,
                 period: int,
                 ensemble_transitions: bool,
                 ensemble_size: int = 5,
                 robust: bool = False):

        super().__init__(alias='QGTS')

        self.n_quantiles = n_quantiles
        self.quantile_on = quantile_on
        self.transition_mats = {}
        self.period = period
        self.robust = robust
        self.ensemble_transitions = ensemble_transitions
        self.uid_pw_distance = {}
        self.ensemble_size = ensemble_size
        self.ensemble_transition_mats = {}

    def transform(self, df: pd.DataFrame, **kwargs):

        # df2 = self.diff(df)

        df_ = df.copy()

        df_ = self.decompose_tsd(df_, period=self.period, robust=self.robust)

        df_['Quantile'] = self._get_quantiles(df_)

        self._calc_transition_matrix(df_)
        if self.ensemble_transitions:
            self.ensemble_transition_mats = self._get_ensemble_transition_mats()

        synth_ts_dict = self._create_synthetic_ts(df_)

        synth_df = self._postprocess_df(df_, synth_ts_dict)

        # df2 = self.undo_diff(df)

        return synth_df

    def _postprocess_df(self, df: pd.DataFrame, synth_ts: Dict):
        synth_list = []
        for uid, uid_df in df.groupby('unique_id'):
            uid_df[self.quantile_on] = synth_ts[uid].values
            synth_list.append(uid_df)

        synth_df = pd.concat(synth_list)

        synth_df['y'] = synth_df[['trend', 'seasonal', 'remainder']].sum(axis=1)
        synth_df = synth_df.drop(columns=['trend', 'seasonal', 'remainder', 'Quantile'])

        synth_df['unique_id'] = synth_df['unique_id'].apply(lambda x: f'{self.alias}_{x}')
        synth_df = synth_df[['ds', 'unique_id', 'y']]

        return synth_df

    def _create_synthetic_ts(self, df: pd.DataFrame) -> Dict:
        quantile_series = self._generate_quantile_series(df)

        generated_time_series = {}

        uids = df['unique_id'].unique().tolist()
        for uid in uids:
            uid_df = df.query(f'unique_id=="{uid}"')
            uid_s = uid_df[self.quantile_on]
            uid_quantiles = uid_df['Quantile']

            uid_q_vals = {q: uid_s[uid_quantiles == q].values for q in range(self.n_quantiles)}

            synth_ts = np.zeros(len(uid_s))
            synth_ts[0] = uid_s.values[0]

            for i in range(1, len(uid_quantiles)):
                current_quantile = quantile_series[uid][i]

                if len(uid_q_vals[current_quantile]) > 0:
                    synth_ts[i] = np.random.choice(uid_q_vals[current_quantile])
                else:
                    synth_ts[i] = np.nan

            generated_time_series[uid] = pd.Series(synth_ts, index=uid_df.index)

        return generated_time_series

    def _generate_quantile_series(self, df: pd.DataFrame):
        uids = df['unique_id'].unique().tolist()

        quantile_series = {}
        for uid in uids:
            if self.ensemble_transitions:
                transition_mat = self.ensemble_transition_mats[uid]
            else:
                transition_mat = self.transition_mats[uid]

            uid_df = df.query(f'unique_id=="{uid}"')

            series = uid_df[self.quantile_on]
            q_series = np.zeros(len(series), dtype=int)

            q_series[0] = uid_df['Quantile'].values[0]  # starts with 1st q

            for t in range(1, len(q_series)):
                current_quantile = q_series[t - 1]

                next_quantile = np.random.choice(np.arange(self.n_quantiles), p=transition_mat[current_quantile])
                q_series[t] = next_quantile

            quantile_series[uid] = q_series

        return quantile_series

    def _get_ensemble_transition_mats(self):
        mats = copy.deepcopy(self.transition_mats)

        uid_pairs = combinations([*mats], 2)

        for uid in mats:
            self.uid_pw_distance[(uid, uid)] = 0.0

        for uid1, uid2 in uid_pairs:
            mat1 = mats[uid1]
            mat2 = mats[uid2]
            dist = np.linalg.norm(mat1 - mat2)

            self.uid_pw_distance[(uid1, uid2)] = dist
            self.uid_pw_distance[(uid2, uid1)] = dist

        ensemble_mats = {}
        for uid in mats:
            uid_dists = pd.Series(
                {other_uid: self.uid_pw_distance[(uid, other_uid)]
                 for other_uid in mats})

            similar_uids = uid_dists.sort_values().head(self.ensemble_size).index.tolist()

            avg_mat = np.sum(
                mats[uid]
                for uid in similar_uids
            ) / self.ensemble_size

            ensemble_mats[uid] = avg_mat

        return ensemble_mats

    def _calc_transition_matrix(self, df: pd.DataFrame):
        assert 'Quantile' in df.columns

        for unique_id, group in df.groupby('unique_id'):
            quantiles = group['Quantile'].values

            t_count_matrix = np.zeros((self.n_quantiles, self.n_quantiles))

            # Loop through the quantiles and count transitions
            for i in range(len(quantiles) - 1):
                current_quantile = quantiles[i]
                next_quantile = quantiles[i + 1]
                t_count_matrix[current_quantile, next_quantile] += 1

            t_prob_matrix = t_count_matrix / t_count_matrix.sum(axis=1, keepdims=True)
            t_prob_matrix = np.nan_to_num(t_prob_matrix)

            # rows where the sum is zero
            for row in range(self.n_quantiles):
                if np.sum(t_count_matrix[row]) == 0:
                    t_prob_matrix[row] = np.ones(self.n_quantiles) / self.n_quantiles
                else:
                    t_prob_matrix[row] = t_count_matrix[row] / np.sum(t_count_matrix[row])

            self.transition_mats[unique_id] = t_prob_matrix

        return self.transition_mats

    def _get_quantiles(self, df: pd.DataFrame):
        assert self.quantile_on in df.columns

        quantiles = df.groupby('unique_id')[self.quantile_on].transform(
            lambda x: pd.qcut(x, self.n_quantiles, labels=False, duplicates='drop')
        )

        return quantiles

    @staticmethod
    def decompose_tsd(df: pd.DataFrame, period: int, robust: bool):
        seasonal_components = []
        trend_components = []
        remainder_components = []

        for unique_id, group in df.groupby('unique_id'):
            stl = STL(group['y'], period=period, robust=robust)
            result = stl.fit()

            seasonal_components.append(pd.DataFrame({
                'unique_id': unique_id,
                'ds': group['ds'],
                'seasonal': result.seasonal
            }))

            trend_components.append(pd.DataFrame({
                'unique_id': unique_id,
                'ds': group['ds'],
                'trend': result.trend
            }))

            remainder_components.append(pd.DataFrame({
                'unique_id': unique_id,
                'ds': group['ds'],
                'remainder': result.resid
            }))

        seasonal_df = pd.concat(seasonal_components)
        trend_df = pd.concat(trend_components)
        remainder_df = pd.concat(remainder_components)

        decomposed_df = pd.merge(seasonal_df, trend_df, on=['unique_id', 'ds'])
        decomposed_df = pd.merge(decomposed_df, remainder_df, on=['unique_id', 'ds'])

        return decomposed_df

class QuantileDerivedTimeSeriesGenerator:
    def __init__(self, n_quantiles: int, ensemble_transitions: bool, ensemble_size: int = 0):
        """
        Initialize the generator.
        
        Args:
            n_quantiles (int): Number of quantiles for binning the differenced series.
            ensemble_transitions (bool): Whether to use ensemble transition matrices.
            ensemble_size (int): Number of similar series to include in the ensemble transition matrix.
        """
        self.n_quantiles = n_quantiles
        self.ensemble_transitions = ensemble_transitions
        self.ensemble_size = ensemble_size
        self.transition_mats = {}
        self.ensemble_transition_mats = {}
        self.uid_pw_distance = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic time series.
        
        Args:
            df (pd.DataFrame): Input time series with columns ['ds', 'unique_id', 'y'].
        
        Returns:
            pd.DataFrame: Synthetic time series with the same structure.
        """
        # Step 1: Calculate differences
        df_diff = self._difference_series(df)

        # Step 2: Quantile binning
        df_diff['Quantile'] = self._get_quantiles(df_diff)

        # Step 3: Transition matrix
        self._calc_transition_matrix(df_diff)
        if self.ensemble_transitions:
            self.ensemble_transition_mats = self._get_ensemble_transition_mats()

        # Step 4: Generate synthetic quantile series
        synth_quantile_series = self._generate_quantile_series(df_diff)

        # Step 5: Generate synthetic differenced series
        synth_diff_dict = self._create_synthetic_diff_series(df_diff, synth_quantile_series)

        # Step 6: Integrate to reconstruct original series
        synth_df = self._integrate_series(df, synth_diff_dict)

        return synth_df

    def _difference_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute first-order differences of the series.
        """
        df_diff = df.copy()
        df_diff['diff'] = df.groupby('unique_id')['y'].diff()
        df_diff.dropna(subset=['diff'], inplace=True)  # Remove rows where diff is NaN
        return df_diff

    def _get_quantiles(self, df: pd.DataFrame) -> pd.Series:
        """
        Divide the differenced series into quantiles.
        """
        return df.groupby('unique_id')['diff'].transform(
            lambda x: pd.qcut(x, self.n_quantiles, labels=False, duplicates='drop')
        )

    def _calc_transition_matrix(self, df: pd.DataFrame):
        assert 'Quantile' in df.columns

        for unique_id, group in df.groupby('unique_id'):
            quantiles = group['Quantile'].values

            t_count_matrix = np.zeros((self.n_quantiles, self.n_quantiles))

            # Loop through the quantiles and count transitions
            for i in range(len(quantiles) - 1):
                current_quantile = quantiles[i]
                next_quantile = quantiles[i + 1]
                t_count_matrix[current_quantile, next_quantile] += 1

            # Normalize the rows to get probabilities
            t_prob_matrix = t_count_matrix / t_count_matrix.sum(axis=1, keepdims=True)
            t_prob_matrix = np.nan_to_num(t_prob_matrix)  # Handle rows with sum zero

            # Ensure rows with all zeros are replaced with uniform probabilities
            for row in range(self.n_quantiles):
                if np.sum(t_count_matrix[row]) == 0:  # Row with no transitions
                    t_prob_matrix[row] = np.ones(self.n_quantiles) / self.n_quantiles

            self.transition_mats[unique_id] = t_prob_matrix

        return self.transition_mats


    def _get_ensemble_transition_mats(self):
        # Copy transition matrices for safety
        mats = copy.deepcopy(self.transition_mats)

        # Initialize pairwise distances dictionary
        self.uid_pw_distance = {}

        # Compute pairwise distances between all unique_ids
        uid_pairs = combinations(mats.keys(), 2)

        for uid1, uid2 in uid_pairs:
            mat1 = mats[uid1]
            mat2 = mats[uid2]
            
            # Calculate Euclidean distance between transition matrices
            dist = np.linalg.norm(mat1 - mat2)
            
            # Store distances bidirectionally
            self.uid_pw_distance[(uid1, uid2)] = dist
            self.uid_pw_distance[(uid2, uid1)] = dist

        # Add zero distance for self comparisons
        for uid in mats:
            self.uid_pw_distance[(uid, uid)] = 0.0

        # Create ensemble transition matrices
        ensemble_mats = {}
        for uid in mats:
            # Get distances of current uid to all others
            uid_dists = pd.Series({
                other_uid: self.uid_pw_distance[(uid, other_uid)]
                for other_uid in mats
            })

            # Select the most similar series based on distance
            similar_uids = uid_dists.nsmallest(self.ensemble_size).index.tolist()

            # Average the transition matrices of similar series
            avg_mat = np.mean([mats[similar_uid] for similar_uid in similar_uids], axis=0)

            # Store the ensemble transition matrix
            ensemble_mats[uid] = avg_mat

        return ensemble_mats


    def _generate_quantile_series(self, df: pd.DataFrame) -> Dict:
        """
        Generate synthetic quantile series using the transition matrix.
        """
        quantile_series = {}
        for uid, group in df.groupby('unique_id'):
            if self.ensemble_transitions:
                transition_mat = self.ensemble_transition_mats[uid]
            else:
                transition_mat = self.transition_mats[uid]

            original_quantiles = group['Quantile'].values
            q_series = np.zeros(len(original_quantiles), dtype=int)

            q_series[0] = original_quantiles[0]  # Start with the first quantile

            for t in range(1, len(q_series)):
                current_quantile = q_series[t - 1]
                next_quantile = np.random.choice(self.n_quantiles, p=transition_mat[current_quantile])
                q_series[t] = next_quantile

            quantile_series[uid] = q_series

        return quantile_series

    def _create_synthetic_diff_series(self, df: pd.DataFrame, quantile_series: Dict) -> Dict:
        """
        Generate synthetic differenced series based on the synthetic quantile series.
        """
        generated_diff_series = {}
        for uid, group in df.groupby('unique_id'):
            quantiles = group['Quantile']
            diff_values = group['diff']
            q_series = quantile_series[uid]

            q_bins = {q: diff_values[quantiles == q].values for q in range(self.n_quantiles)}

            synth_diff = np.zeros(len(q_series))
            for i, q in enumerate(q_series):
                if len(q_bins[q]) > 0:
                    synth_diff[i] = np.random.choice(q_bins[q])
                else:
                    synth_diff[i] = 0  # Fallback to 0 if bin is empty

            generated_diff_series[uid] = pd.Series(synth_diff, index=group.index)

        return generated_diff_series

    def _integrate_series(self, df: pd.DataFrame, synth_diff_dict: Dict) -> pd.DataFrame:
        """
        Integrate the synthetic differenced series to reconstruct the original scale.
        """
        synth_list = []
        for uid, group in df.groupby('unique_id'):
            original_y = group['y'].iloc[0]  # Starting value
            synth_diff = synth_diff_dict[uid]

            synthetic_y = np.cumsum(np.insert(synth_diff.values, 0, original_y))
            group['y'] = synthetic_y
            synth_list.append(group)

        synth_df = pd.concat(synth_list)
        return synth_df[['ds', 'unique_id', 'y']]
