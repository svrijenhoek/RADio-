import math

import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer


class Divergence:
    """
    Class that calculates the divergence between two distributions P and Q.
    Assumes two dictionaries with the same keys as input
    """

    def __init__(self, metric='JSD'):
        self.metric = metric

    def opt_merge_max_mappings(self, a, b):
        """ Merges two dictionaries based on the largest value in a given mapping.
        Parameters
        ----------
        distr_pool : Dict[Any, Comparable]
        distr_recommendation : Dict[Any, Comparable]
        Returns
        -------
        Dict[Any, Comparable]
            The merged dictionary
        """
        merged, other = (a, b) if len(a) > len(
            b) else (b, a)
        merged = dict(merged)
        for key in other:
            if key not in merged or other[key] > merged[key]:
                merged[key] = other[key]
        return merged

    def compute(self, s, q, alpha=0.0001):
        """
        KL (p || q), the lower the better.
        alpha is not really a tuning parameter, it's just there to make the
        computation more numerically stable.
        """
        try:
            assert 0.99 <= sum(s.values()) <= 1.01
            assert 0.99 <= sum(q.values()) <= 1.01
        except AssertionError:
            print("Input not normalized distributions with matching keys")
            print(sum(s.values()))
            print(sum(q.values()))
            pass

        ss = []
        qq = []
        merged_dic = self.opt_merge_max_mappings(q, s)
        for key in sorted(merged_dic.keys()):
            recom_score = s.get(key, 0.)
            pool_score = q.get(key, 0.)
            qq.append((1 - alpha) * pool_score + alpha * recom_score)
            ss.append((1 - alpha) * recom_score + alpha * pool_score)
        if self.metric == 'JSD':
            return self.JSD(ss, qq)
        elif self.metric == 'KL':
            return entropy(ss, qq, base=2)

    @staticmethod
    def JSD(P, Q):
        """ Compute J-S divergence.
            Parameters
            ----------
            P : Dictionary. distribution of pool.
            Q : Dictionary. distribution of recommendation.
            Returns
            -------
            JS divergence value.

        """
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        try:
            jsd_root = math.sqrt(
                abs(0.5 * (entropy(_P, _M, base=2) + entropy(_Q, _M, base=2))))
        except ZeroDivisionError:
            jsd_root = None
        return jsd_root


class DistributionBuilder:
    """
    Class that turns a list of properties into a normalized distribution
    """

    def __init__(self, feature_type, discount, **kwargs):
        self.feature_type = feature_type
        self.discount = discount

        if self.feature_type == 'cont':
            bins = kwargs.get('bins', 10)
            self.bins_discretizer = KBinsDiscretizer(
                encode='ordinal', n_bins=bins, strategy='uniform', subsample=None)

    def build_distribution(self, x):
        if self.feature_type == 'cat':
            return self.categorical(x)
        elif self.feature_type == 'cont':
            return self.continuous(x)
        elif self.feature_type == 'cat_m':
            return self.categorical_multi(x)

    @staticmethod
    def harmonic_number(n):
        """Returns an approximate value of n-th harmonic number.
        http://en.wikipedia.org/wiki/Harmonic_number
        """
        # Euler-Mascheroni constant
        gamma = 0.57721566490153286060651209008240243104215933593992
        return gamma + math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

    def categorical(self, x):
        """"
        Parameters
        ----------
        x : List of properties, where the first entry refers to the first article, the second entry to the second, etc.
        r : Boolean. Should the distribution be discounted or not
        Returns
        -------
        Dictionary where every entry refers to the presence of that property in the distribution.

        """
        n = len(x)
        sum_one_over_ranks = self.harmonic_number(n)
        count = 0
        distribution = {}
        for i, item in enumerate(x):
            rank = i + 1
            feature_freq = distribution.get(item, 0.)
            distribution[item] = feature_freq + 1 * 1 / rank / \
                                 sum_one_over_ranks if self.discount else feature_freq + 1 * 1 / n
            count += 1
        return distribution

    def categorical_multi(self, x):
        """"
        Parameters
        ----------
        x : List of properties, where the first entry refers to the first article, the second entry to the second, etc.
        r : Boolean. Should the distribution be discounted or not
        Returns
        -------
        Dictionary where every entry refers to the presence of that property in the distribution.

        """
        n = len(x)
        sum_one_over_ranks = self.harmonic_number(n)
        distribution = {}
        for i, item in enumerate(x):
            for j, entry in enumerate(item):
                rank = i + 1
                feature_freq = distribution.get(entry, 0.)
                distribution[entry] = feature_freq + 1 * 1 / rank / \
                                     sum_one_over_ranks if self.discount else feature_freq + 1 * 1 / n

        if len(distribution) > 0:
            factor = 1.0 / sum(distribution.values())
            for k in distribution:
                distribution[k] = distribution[k] * factor
            return distribution
        else:
            return None

    def continuous(self, x):
        n = len(x)
        sum_one_over_ranks = self.harmonic_number(n)
        arr_binned = self.bins_discretizer.transform(np.array(x).reshape(-1, 1))
        distribution = {}
        if self.discount:
            for bin in list(range(self.bins_discretizer.n_bins)):
                for i, ele in enumerate(arr_binned[:, 0]):
                    if ele == bin:
                        rank = i + 1
                        bin_freq = distribution.get(bin, 0.)
                        distribution[bin] = bin_freq + 1 * 1 / rank / sum_one_over_ranks
        else:
            for bin in list(range(self.bins_discretizer.n_bins)):
                distribution[bin] = round(np.count_nonzero(
                    arr_binned == bin) / arr_binned.shape[0], 3)
        return distribution


class Metric:

    def __init__(self, **kwargs):
        self.feature_type = kwargs.get('feature_type', 'cat')
        self.rank_aware_recommendation = kwargs.get('rank_aware_recommendation', True)
        self.rank_aware_context = kwargs.get('rank_aware_context', True)
        self.bins = kwargs.get('bins', 10)
        self.context_type = kwargs.get('context', 'dynamic')

        self.divergence = Divergence(metric=kwargs.get('metric', 'JSD'))

        self.recommendation_builder = DistributionBuilder(self.feature_type, self.rank_aware_recommendation)
        self.context_builder = DistributionBuilder(self.feature_type, self.rank_aware_context)

        self.Q_static = {}

    def compute(self, recommendation, context):

        if self.context_type == 'dynamic':
            Q = self.context_builder.build_distribution(context)
        else:
            if not self.Q_static:
                if self.feature_type == 'cont':
                    self.context_builder.bins_discretizer.fit(np.array(context).reshape(-1, 1))
                    self.recommendation_builder.bins_discretizer.fit(np.array(context).reshape(-1, 1))
                self.Q_static = self.context_builder.build_distribution(context)
            Q = self.Q_static

        P = self.recommendation_builder.build_distribution(recommendation)


        if P and Q:
            return self.divergence.compute(P, Q)
        else:
            return None
