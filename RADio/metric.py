import numpy as np

from RADio.divergence import Divergence
from RADio.distributions import DistributionBuilder


class DiversityMetric:

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
