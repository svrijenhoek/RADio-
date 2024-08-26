import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class DistributionBuilder:
    """
    Class that turns a list of properties into a normalized distribution
    feature_type: [categorical (cat), categorical_multi (cat_m), continuous (cont)]
    rank_aware: should the values be discounted based on their rank in the recommendation
    bins: optional - into how many bins should continuous values be divided
    """

    def __init__(self, feature_type, rank_aware, **kwargs):
        self.feature_type = feature_type
        self.rank_aware = rank_aware

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
        distribution = {}
        count = 0
        for _, item in enumerate(x):
            count += 1
            feature_freq = distribution.get(item, 0.)
            distribution[item] = feature_freq + 1 * 1 / count / \
                                 sum_one_over_ranks if self.rank_aware else feature_freq + 1 * 1 / n
        return distribution

    def categorical_multi(self, x):
        """"
        Build distributions where the relevant feature can have multiple values. For example, multiple people can be
        mentioned in an article, or an article can have multiple topics/categories assigned.
        Parameters
        ----------
        x : List of properties, where the first entry refers to the first article, the second entry to the second, etc.
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
                                     sum_one_over_ranks if self.rank_aware else feature_freq + 1 * 1 / n
        # normalizing the distribution is a bit harder when it's unknown how many entities should be accounted for
        if len(distribution) > 0:
            factor = 1.0 / sum(distribution.values())
            for k in distribution:
                distribution[k] = distribution[k] * factor
            return distribution
        else:
            return None

    def continuous(self, x):
        """"
        List of continuous values. Since the divergence-based metric is essentially categorical, these values are first
        binned. This means that we lose information about ordering, and could use improvement in the future.
        Parameters
        ----------
        x : List of properties, where the first entry refers to the first article, the second entry to the second, etc.
        Returns
        -------
        Dictionary where every entry refers to the presence of that property in the distribution.

        """
        n = len(x)
        sum_one_over_ranks = self.harmonic_number(n)
        arr_binned = self.bins_discretizer.transform(np.array(x).reshape(-1, 1))
        distribution = {}
        if self.rank_aware:
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
