import math
from numpy.linalg import norm
from scipy.stats import entropy


class Divergence:
    """
    Class that calculates the divergence between two distributions P and Q.
    Assumes two dictionaries with the same keys as input
    """

    def __init__(self, metric='JSD'):
        self.metric = metric

    @staticmethod
    def opt_merge_max_mappings(a, b):
        """ Merges two dictionaries based on the largest value in a given mapping.
        Parameters
        ----------
        a : Dict[Any, Comparable]
        b : Dict[Any, Comparable]
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

    def compute(self, s, q, alpha=0.001):
        """
        KL (p || q), the lower the more similar the distributions are.
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
            s_score = s.get(key, 0.)
            q_score = q.get(key, 0.)
            qq.append((1 - alpha) * s_score + alpha * q_score)
            ss.append((1 - alpha) * q_score + alpha * s_score)
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
