"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi
"""

import concurrent.futures
# import copy
import collections
import json
import os

import numpy as np
import pandas as pd
from utils import logger

import constants

class ImportantFeatureSelector(object):
    def __init__(self, feature_importance_df, criteria, fixed_features=None):
        """
        :param feature_importance_df A Dataframe with FeatureName and Importance columns; index is feature ID (0-(2351-1))
        :param criteria: Determines which features to return.
            Possible values:
                'most_important' - The most important N features will be returned.
                'least_important' - The least important N features will be returned.
                'least_important_nz' - The least important N features with non-zero importance will be returned.
        """
        self.feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False)
        self.criteria = criteria
        self.fixed_features = fixed_features
        self.criteria_desc_map = {'most_important': '(most_imp) Features chosen from most important N features',
                                  'least_important': '(least_imp) Features chosen from least important N features',
                                  'least_important_nz': '(least_imp_nz) Features chosen from least important N non-zero features',
                                  'median_important': '(median_imp) Choses N features around the median importance value',
                                  }

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_features(self, num_features):
        if self.criteria.startswith('most_important'):
            most_important = self.feature_importance_df.index
            most_important = most_important[most_important.isin(
                self.fixed_features)]
            result = most_important[:num_features]

            # result = self.feature_importance_df.head(
            #    num_features).index[:num_features]

        elif self.criteria == 'least_important':
            result = self.feature_importance_df.tail(
                num_features).index[:num_features]
        elif self.criteria == 'least_important_nz':
            temp = self.feature_importance_df[self.feature_importance_df['Importance'] != 0]
            result = temp.tail(num_features).index[:num_features]
        elif self.criteria == 'median_important':
            median_index = self.feature_importance_df.shape[0] // 2
            result = self.feature_importance_df.index[median_index -
                                                      num_features // 2:]
            result = result[:num_features]
        else:
            raise ValueError(
                'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return list(result)


class ShapleyFeatureSelector(object):
    def __init__(self, shap_values_df, criteria, fixed_features=None):
        """ Picks features based on the sum of Shapley values across all samples.
        :param shap_values_df A Dataframe of shape <# samples> x <# features>
        :param criteria: Determines which features to return.
            Possible values:
                'shap_smallest' - Returns the N features whose summed Shapley values are smallest (including negative).
                'shap_largest' - Returns the N features whose summed Shapley values are largest.
                'shap_nearest_zero' - Returns the N features whose summed Shapley values are closest to zero.
                'shap_nearest_zero_nz' - Returns the N features whose summed Shapley values are closest to zero and also not zero.
                'shap_nearest_zero_nz_abs' - Returns the N features whose summed absolute Shapley values are closest to zero and also not zero.
        """
        self.shap_values_df = shap_values_df
        self.criteria = criteria
        self.fixed_features = fixed_features
        self.criteria_desc_map = {'shap_smallest': '(shap_smallest) Choses N features whose summed Shapley values are smallest (including negative).',
                                  'shap_largest': '(shap_smallest) Choses N features whose summed Shapley values are largest.',
                                  'shap_nearest_zero': '(shap_nearest_zero) Choses N features whose summed Shapley values are closest to zero',
                                  'shap_nearest_zero_nz': '(shap_nearest_zero_nz) Choses N features whose summed Shapley values are closest to zero but not zero',
                                  'shap_nearest_zero_nz_abs': '(shap_nearest_zero_nz_abs) Choses N features whose summed absolute Shapley values are closest to zero but not zero'}

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_features(self, num_features):
        if self.criteria == 'shap_nearest_zero':
            summed = self.shap_values_df.sum()
            closest_to_zero = summed.abs().argsort()
            result = list(closest_to_zero[:num_features])
        elif self.criteria == 'shap_smallest':
            summed = self.shap_values_df.sum()
            result = list(summed.argsort()[:num_features])
        elif self.criteria == 'shap_largest':
            summed = self.shap_values_df.sum()
            result = list(summed.argsort()[-num_features:])
        elif self.criteria == 'shap_nearest_zero_nz':
            summed = self.shap_values_df.sum()
            summed[summed == 0.0] = np.inf
            closest_to_zero = summed.abs().argsort()
            result = list(closest_to_zero[:num_features])
        elif self.criteria == 'shap_nearest_zero_nz_abs':
            summed = self.shap_values_df.abs().sum()
            summed[summed == 0.0] = np.inf
            closest_to_zero = summed.argsort()
            result = list(closest_to_zero[:num_features])
        elif self.criteria == 'fixed_shap_nearest_zero_nz_abs':
            summed = self.shap_values_df.abs().sum()
            summed[summed == 0.0] = np.inf
            closest_to_zero = summed.argsort()
            # temp_features = [620, 618]
            # closest_to_zero = closest_to_zero[closest_to_zero.isin(temp_features)]
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            result = list(closest_to_zero[:num_features])
            # result = list(closest_to_zero)
            logger.info(result)
        elif self.criteria.startswith('shap_largest_abs'):
            summed = self.shap_values_df.abs().sum()
            closest_to_zero = summed.argsort()
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            result = list(closest_to_zero[-num_features:])
            logger.info(result)
        else:
            raise ValueError(
                'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return result


class HistogramBinValueSelector(object):
    def __init__(self, criteria, bins):
        """
        :param X: The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        :param criteria: Determines which bucket the returned values will fall in.
            Example: For illustrative purposes consider `bins`=5 and
                `X`=[[0, 1]
                     [0, 2],
                     [0, 2],
                     [0, 3],
                     [0, 3],
                     [0, 3],
                     [0, 4],
                     [0, 4],
                     [0, 5],
                     [0, 5]]
            Possible values:
                'min_bucket' - The bucket with the smallest overall value is chosen. Bucket 0 in the example.
                'max_bucket' - The bucket with the largest overall value is chosen. Bucket 4 in the example.
                'max_population' - The bucket with the most hits. Bucket 2 in the example.
                'min_population' - The bucket with the fewest hits. Bucket 0 in the example.
        :param bins: The number of bins to divide the samples into.
        """
        self.criteria = criteria
        self.criteria_desc_map = {'min_bucket': '(minval) Values chosen from bin with smallest value',
                                  'max_bucket': '(maxval) Values chosen from bin with largest value',
                                  'max_population': '(maxpop) Values chosen from bin with the largest population',
                                  'min_population': '(minpop) Values chosen from bin with the smallest population',
                                  }
        self.histogram_cache = {}
        self.bins = bins

        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    def get_feature_values(self, feature_ids):
        result = []
        for feature_id in feature_ids:
            count = collections.Counter(self._X[:, feature_id])
            result.append(min(count, key=count.get))
        # for feature_id in feature_ids:
        #    if feature_id not in self.histogram_cache:
        #        self.histogram_cache[feature_id] = np.histogram(
        #            self._X[:, feature_id], bins=self.bins)
        #    histogram_bucket_counts = self.histogram_cache[feature_id][0]
        #    histogram_edge_values = self.histogram_cache[feature_id][1]
        #    if self.criteria == 'min_bucket':
        #        bucket_index = 0
        #    elif self.criteria == 'max_bucket':
        #        bucket_index = len(histogram_bucket_counts) - 1
        #    elif self.criteria == 'max_population':
        #        bucket_index = np.argmax(histogram_bucket_counts)
        #    elif self.criteria == 'min_population':
        #        bucket_index = np.argmin(histogram_bucket_counts)
        #    else:
        #        raise ValueError(
        #            'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        #    result.append(
        #        np.mean(histogram_edge_values[bucket_index:bucket_index + 2]))

        return result


def _process_one_shap_linear_combination(feature_index_id_x_shaps_tuple):
    feat_index = feature_index_id_x_shaps_tuple[0]
    feature_id = feature_index_id_x_shaps_tuple[1]
    features_sample_values = feature_index_id_x_shaps_tuple[2]
    this_features_abs_inverse_shaps = feature_index_id_x_shaps_tuple[3]
    alpha = feature_index_id_x_shaps_tuple[4]
    beta = feature_index_id_x_shaps_tuple[5]

    # First, find values and how many times they occur
    (values, counts) = np.unique(features_sample_values, return_counts=True)
    counts = np.array(counts)
    # logger.info('# Feature {} has {} unique values'.format(feature_id, len(counts)))
    sum_abs_shaps = np.zeros(len(values))
    for i in range(len(values)):
        desired_values_mask = features_sample_values == values[i]
        sum_abs_shaps[i] = np.sum(
            desired_values_mask * this_features_abs_inverse_shaps)
    sum_abs_shaps = alpha * (1.0 / counts) + beta * sum_abs_shaps
    values_index = np.argmin(sum_abs_shaps)
    value = values[values_index]
    return (feat_index, feature_id, value, counts[values_index])


def _process_one_shap_value_selection(feature_index_id_x_inv_shaps_tuple):
    feat_index = feature_index_id_x_inv_shaps_tuple[0]
    feature_id = feature_index_id_x_inv_shaps_tuple[1]
    features_sample_values = feature_index_id_x_inv_shaps_tuple[2]
    this_features_abs_inverse_shaps = feature_index_id_x_inv_shaps_tuple[3]
    multiply_by_counts = feature_index_id_x_inv_shaps_tuple[4]

    #                               n
    #                              __
    #                              \
    # <chosen value> = argmax Nv * /  1 / | Sij(v) |
    #                              --
    #                               i = 0
    # Basically, it is just saying, take the value with the highest weight, where weight is based on the
    # number of times that value occurred in that dimension (N_v) and the SHAP values for that value (S_i,j(v))
    #
    # The |S_i,j(v)| component is basically saying that we want to consider SHAP values that are either
    # extremely biased toward malware (+1) or goodware (-1) to be equally 'bad'.
    #
    # The inverse (1/S_i,j(v)) is trying to capture that we generally prefer values whose SHAP value are closer
    # to zero, as better than those that are large.
    #
    # The summation accumulates those SHAP-oriented weights over all the samples that the value occurred in.
    #
    # N_v is the simple count of the number of times that the value v occurred for this dimension across all
    # the samples.
    #
    # First, find values and how many times they occur
    (values, counts) = np.unique(features_sample_values, return_counts=True)
    counts = np.array(counts)
    # logger.info('# Feature {} has {} unique values'.format(feature_id, len(counts)))
    sum_inverse_abs_shaps = np.zeros(len(values))
    for i in range(len(values)):
        desired_values_mask = features_sample_values == values[i]
        sum_inverse_abs_shaps[i] = np.sum(
            desired_values_mask * this_features_abs_inverse_shaps)
    if multiply_by_counts:
        sum_inverse_abs_shaps = counts * sum_inverse_abs_shaps
    values_index = np.argmax(sum_inverse_abs_shaps)
    value = values[values_index]
    return (feat_index, feature_id, value, counts[values_index])


class ShapValueSelector(object):
    def __init__(self, shaps_for_x, criteria, cache_dir=None):
        """
        Selects feature values by looking at the Shapley values for the samples' features.
        :param shaps_for_x: The Shapley values for all samples of X. NxM where N = num samples, M = num features
        :param criteria: .
            Possible values:
                'argmax_Nv_sum_inverse_shap' - argmax(Nv * summation over samples(1 / Shap[sample, feature]).
                'argmax_sum_inverse_shap' - argmax(summation over samples(1 / Shap[sample, feature]).
                    (This is the same as `argmax_Nv_sum_inverse_shap` but with no multiply by Nv operation
        :param cache_dir: A path to a directory where cached calculated values will be stored.
            If a cache file for this criteria exists when the instance is created it's cache will be warmed with the
            values from this file. This cache file is used since the calculations behind selecting the values is very
            expensive while being invariant from run to run.
        """
        self.shaps_for_x = shaps_for_x
        self.criteria = criteria
        self.criteria_desc_map = {'argmax_Nv_sum_inverse_shap': 'argmax(Nv * sum(1/Shap[sample,feature]))',
                                  'argmax_sum_inverse_shap': 'argmax(sum(1/Shap[sample,feature])',
                                  'argmin_Nv_sum_abs_shap': 'argmin(1/count + sum(abs(Shap[sample,feature])))',
                                  'argmin_sum_abs_shap': 'argmin(sum(abs(Shap[sample,feature])))'}

        # Calculate 1 / shap values now since we use those values often
        # This can result in a RuntimeWarning: divide by zero encountered in true_divide
        # But it is safe to ignore that warning in this case.
        self.abs_shaps = abs(self.shaps_for_x)
        if self.criteria.startswith('argmax'):
            self.inverse_abs_shaps = 1.0 / abs(self.abs_shaps)
            self.inverse_abs_shaps[self.inverse_abs_shaps == np.inf] = 0

        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

        self._cache = {}
        self.cache_file = os.path.join(
            cache_dir, criteria + '.json') if cache_dir else None
        self._load_cache()

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        # We don't invalidate self._cache here since we make an assumption that when X gets reset
        # that it is set to the same X as before. Thus we get the benefit of caching these values
        # across experiment iterations.
        if self._X is not None:
            assert self._X.shape == value.shape
        self._X = value

    def _load_cache(self):
        if self.cache_file and os.path.isfile(self.cache_file):
            with open(self.cache_file, 'r') as f:
                temp = json.load(f)
            # json serializes keys as strings so convert to ints as we expected
            self._cache = {int(key): value for key, value in temp.items()}

    def _save_cache(self):
        if self.cache_file:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=4)

    def get_feature_values(self, feature_ids):
        result = [0] * len(feature_ids)
        if self.criteria == 'argmax_Nv_sum_inverse_shap' or self.criteria == 'argmax_sum_inverse_shap':
            multiply_by_counts = self.criteria == 'argmax_Nv_sum_inverse_shap'
            to_be_calculated = []
            for feat_index, feature_id in enumerate(feature_ids):
                if feature_id in self._cache:
                    result[feat_index] = self._cache[feature_id]
                else:
                    to_be_calculated.append(
                        (feat_index, feature_id, self._X[:, feature_id], self.inverse_abs_shaps[:, feature_id], multiply_by_counts))
            if len(to_be_calculated) != 0:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    map_result = executor.map(
                        _process_one_shap_value_selection, to_be_calculated)
                    for (feat_index, feature_id, value, samples_with_value_count) in list(map_result):
                        if constants.VERBOSE:
                            logger.info('Selected value {} for feature {}. {} samples have that same value'.format(value,
                                                                                                             feature_id,
                                                                                                             samples_with_value_count))
                        result[feat_index] = value
                        self._cache[int(feature_id)] = float(value)
            self._save_cache()
        elif self.criteria.startswith('argmin'):
            # Controls weighting in linear combo of count and SHAP
            if self.criteria == 'argmin_Nv_sum_abs_shap':
                alpha = 1.0
                beta = 1.0
            else:
                alpha = 0.0
                beta = 1.0
            to_be_calculated = []
            for feat_index, feature_id in enumerate(feature_ids):
                if feature_id in self._cache:
                    result[feat_index] = self._cache[feature_id]
                else:
                    to_be_calculated.append(
                        (feat_index, feature_id, self._X[:, feature_id], self.abs_shaps[:, feature_id], alpha, beta))
            if len(to_be_calculated) != 0:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    map_result = executor.map(_process_one_shap_linear_combination, to_be_calculated)
                    for (feat_index, feature_id, value, samples_with_value_count) in list(map_result):
                        if constants.VERBOSE:
                            logger.info('Selected value {} for feature {}. {} samples have that same value'.format(value,
                                                                                                             feature_id,
                                                                                                             samples_with_value_count))
                        result[feat_index] = value
                        self._cache[int(feature_id)] = float(value)
        else:
            raise ValueError(
                'Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return result


class CombinedShapSelector(object):
    def __init__(self, shap_values_df, criteria, fixed_features=None):
        """
        Selects feature values by looking at the Shapley values for the samples' features.
        :param shaps_for_x: The Shapley values for all samples of X. NxM where N = num samples, M = num features
        :param criteria: .
            Possible values:
                'argmax_Nv_sum_inverse_shap' - argmax(Nv * summation over samples(1 / Shap[sample, feature]).
                'argmax_sum_inverse_shap' - argmax(summation over samples(1 / Shap[sample, feature]).
                    (This is the same as `argmax_Nv_sum_inverse_shap` but with no multiply by Nv operation
        :param cache_dir: A path to a directory where cached calculated values will be stored.
            If a cache file for this criteria exists when the instance is created it's cache will be warmed with the
            values from this file. This cache file is used since the calculations behind selecting the values is very
            expensive while being invariant from run to run.
        """
        self.shap_values_df = shap_values_df
        self.criteria = criteria
        self.criteria_desc_map = {
            'combined_shap': 'shap_largest_abs, argmin(1/count + sum(abs(Shap[sample,feature])))'}
        self.fixed_features = fixed_features
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        # We don't invalidate self._cache here since we make an assumption that when X gets reset
        # that it is set to the same X as before. Thus we get the benefit of caching these values
        # across experiment iterations.
        if self._X is not None:
            assert self._X.shape == value.shape
        self._X = value

    def get_feature_values(self, num_feats, alpha=1.0, beta=1.0):
        selected_features = []
        selected_values = []
        local_X = self._X
        local_shap = self.shap_values_df
        for i in range(num_feats):
            # Get ordered features by largest SHAP
            # summed = local_shap.abs().sum()

            # Get features that are most goodware-leaning
            summed = local_shap.sum()
            closest_to_zero = summed.argsort()
            # Only look at features we care about
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            # Remove features that we have already selected
            closest_to_zero = closest_to_zero[~closest_to_zero.isin(
                selected_features)]
            feature_id = closest_to_zero.iloc[0]
            selected_features.append(feature_id)

            # Run value selection on that dimension
            features_sample_values = local_X[:, feature_id]
            (values, counts) = np.unique(
                features_sample_values, return_counts=True)
            counts = np.array(counts)
            sum_abs_shaps = np.zeros(len(values))
            for j in range(len(values)):
                desired_values_mask = features_sample_values == values[j]
                # import IPython
                # IPython.embed()
                sum_abs_shaps[j] = sum(
                    desired_values_mask * local_shap.values[:, feature_id])
                # ----- Below is from when we are taking absolute value -----
                # sum_abs_shaps[j] = np.sum(abs(
                #    desired_values_mask * local_shap.values[:, feature_id]))
            sum_abs_shaps = alpha * (1.0 / counts) + beta * sum_abs_shaps
            values_index = np.argmin(sum_abs_shaps)
            value = values[values_index]
            selected_values.append(value)
            logger.info(f"i: {i}, feature_id: {feature_id}, value: {value}")

            # Filter data based on existing values
            selection_mask = local_X[:, feature_id] == value
            # import IPython
            # IPython.embed()
            logger.info(local_X[selection_mask].shape)
            local_X = local_X[selection_mask]
            local_shap = local_shap[selection_mask]
        return selected_features, selected_values


class FixedFeatureAndValueSelector(object):
    def __init__(self, feature_value_map):
        """
        Caller passes in specific features, values to be returned. This is most useful for reproducing or debugging
        purposes, not research purposes.
        :param feature_value_map: The (feature ID, feature value) pairs as a dict
        """
        self.feature_value_map = feature_value_map
        self.criteria = 'fixed'
        self.criteria_desc_map = { 'fixed': 'fixed - Caller specifies features, values' }
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        if self._X is not None:
            assert self._X.shape == value.shape
        self._X = value

    def get_features(self, num_features):
        result = list(self.feature_value_map.keys())[:num_features]
        return result

    def get_feature_values(self, features):
        result = [self.feature_value_map[feat] for feat in features]
        return result


class FixedFeatureSelector(object):
    def __init__(self, fixed_feature_list, criteria):
        """
        :param fixed_feature_list: A list of fixed feature identifiers to return
        :param criteria: criterion behind the fixed features
        """
        self.fixed_features = fixed_feature_list
        self.criteria = criteria
        self.criteria_desc_map = {'fixed' 'All the fixed set of features.'}

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    def get_features(self, num_features):
        if self.criteria == 'fixed':
            result = self.fixed_features
        else:
            raise ValueError('Unsupported value of {} for the "criteria" argument'.format(self.criteria))
        return result


class CombinedAdditiveShapSelector(object):
    def __init__(self, shap_values_df, criteria, fixed_features=None):
        """
        Selects feature values by looking at the Shapley values for the samples' features.
        :param shaps_for_x: The Shapley values for all samples of X. NxM where N = num samples, M = num features
        :param criteria: .
            Possible values:
                'argmax_Nv_sum_inverse_shap' - argmax(Nv * summation over samples(1 / Shap[sample, feature]).
                'argmax_sum_inverse_shap' - argmax(summation over samples(1 / Shap[sample, feature]).
                    (This is the same as `argmax_Nv_sum_inverse_shap` but with no multiply by Nv operation
        :param cache_dir: A path to a directory where cached calculated values will be stored.
            If a cache file for this criteria exists when the instance is created it's cache will be warmed with the
            values from this file. This cache file is used since the calculations behind selecting the values is very
            expensive while being invariant from run to run.
        """
        self.shap_values_df = shap_values_df
        self.criteria = criteria
        self.criteria_desc_map = {
            'combined_additive_shap': 'shap_largest_abs, argmin(1/count + sum(abs(Shap[sample,feature])))'}
        self.fixed_features = fixed_features
        # The feature values for all of the samples as a 2 dimensional array (N samples x M features).
        self._X = None

    @property
    def name(self):
        return self.criteria

    @property
    def description(self):
        return self.criteria_desc_map[self.criteria]

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        # We don't invalidate self._cache here since we make an assumption that when X gets reset
        # that it is set to the same X as before. Thus we get the benefit of caching these values
        # across experiment iterations.
        if self._X is not None:
            assert self._X.shape == value.shape
        self._X = value

    def get_feature_values(self, num_feats, alpha=1.0, beta=1.0):
        selected_features = []
        selected_values = []
        local_X = self._X
        local_shap = self.shap_values_df

        # Check the local_x values. If there are zero values, assign a large
        # positive shap contribution to the respective entry in the SHAP matrix
        logger.info(self._X.shape)
        high = 100
        local_shap_np = np.copy(local_shap.to_numpy())
        for i in range(local_X.shape[0]):
            vec = local_X[i]
            zs = np.where(vec == 0)
            local_shap_np[i][zs] = high

        # Now update the local SHAP DataFrame
        local_shap = pd.DataFrame(local_shap_np)

        for i in range(num_feats):
            # Get ordered features by largest SHAP
            # summed = local_shap.abs().sum()

            # Get features that are most goodware-leaning
            summed = local_shap.sum()
            closest_to_zero = summed.argsort()
            # Only look at features we care about
            closest_to_zero = closest_to_zero[closest_to_zero.isin(
                self.fixed_features)]
            # Remove features that we have already selected
            closest_to_zero = closest_to_zero[~closest_to_zero.isin(
                selected_features)]
            feature_id = closest_to_zero.iloc[0]
            selected_features.append(feature_id)

            # Run value selection on that dimension
            features_sample_values = local_X[:, feature_id]
            (values, counts) = np.unique(
                features_sample_values, return_counts=True)
            counts = np.array(counts)
            sum_abs_shaps = np.zeros(len(values))
            for j in range(len(values)):
                desired_values_mask = features_sample_values == values[j]
                sum_abs_shaps[j] = np.sum(
                    desired_values_mask * local_shap.values[:, feature_id])
                # ----- Below is from when we are taking absolute value -----
                # sum_abs_shaps[j] = np.sum(abs(
                #    desired_values_mask * local_shap.values[:, feature_id]))
            sum_abs_shaps = alpha * (1.0 / counts) + beta * sum_abs_shaps
            values_index = np.argmin(sum_abs_shaps)
            value = values[values_index]
            selected_values.append(value)
            logger.info(i, feature_id, value, np.min(sum_abs_shaps))

            # Filter data based on existing values
            selection_mask = local_X[:, feature_id] == value
            logger.info(local_X[selection_mask].shape)
            local_X = local_X[selection_mask]
            local_shap = local_shap[selection_mask]
        return selected_features, selected_values
