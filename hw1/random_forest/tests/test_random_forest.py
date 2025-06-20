from unittest import TestCase

import numpy as np

import sys
sys.path.extend([r"C:\Users\62174\OneDrive\Рабочий стол\Jupyter Notebooks\AIMasters\2 sem\ML2"])

from itertools import chain, cycle

from sem_dt_rf.random_forest.sampler import BaseSampler, ObjectSampler, FeatureSampler
from sem_dt_rf.random_forest.random_forest import RandomForest


class RandomClassifier:
    def __init__(self, n_classes=5, random_state=None):
        self.n_classes = n_classes
        self.random_state = np.random.RandomState(random_state)
        self.X = None
        self.y = None

    def fit(self, x, y):
        self.X, self.y = x, y
        return self

    def predict_proba(self, x):
        if self.X is None or self.y is None:
            raise RuntimeError('Classifier is not fitted')
        return self.random_state.random(size=(len(x), self.n_classes))

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)


class PredefinedSampler(BaseSampler):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices
        self.indices_yield = cycle(self.indices)

    def sample_indices(self, n_objects):
        return next(self.indices_yield)

    def sample(self, x, y=None):
        indices = self.sample_indices(len(x))
        if y is None:
            return x[indices]
        return x[indices], y[indices]


class RandomForestTests(TestCase):
    def test_common_scenario(self):
        object_sampler = ObjectSampler(max_samples=0.7)
        feature_sampler = FeatureSampler(max_samples=0.9)

        n_classes = 3

        rf = RandomForest(
            base_estimator=RandomClassifier,
            object_sampler=object_sampler,
            feature_sampler=feature_sampler,
            n_estimators=10,
            n_classes=n_classes,
        )

        x = np.arange(24).reshape(6, 4)
        y = np.random.choice(n_classes, size=len(x), replace=True)

        rf = rf.fit(x, y)

        self.assertEqual(rf.predict_proba(x).shape, (len(x), n_classes))
        self.assertEqual(rf.predict(x).shape, (len(x),))

    def test_fitted_sub_samples(self):
        object_sampler = PredefinedSampler([
            [0, 1, 2, 3],
            [3, 2, 2, 1],
            [4, 0, 1, 1],
            [1, 2, 2, 3],
        ])

        feature_sampler = PredefinedSampler([
            [2, 3, 3],
            [0, 1, 2],
            [3, 2, 1],
            [3, 0, 1],
        ])

        self.assertEqual(len(object_sampler.indices), len(feature_sampler.indices))
        n_estimators = len(feature_sampler.indices)

        rf = RandomForest(
            base_estimator=RandomClassifier,
            object_sampler=object_sampler,
            feature_sampler=feature_sampler,
            n_estimators=n_estimators,
        )

        n_objects = max(chain.from_iterable(object_sampler.indices), default=-1) + 1
        n_features = max(chain.from_iterable(feature_sampler.indices), default=-1) + 1

        x = np.arange(n_objects * n_features).reshape(n_objects, n_features)
        y = np.arange(len(x))

        rf = rf.fit(x, y)

        self.assertEqual(len(rf.estimators), n_estimators)
        self.assertEqual(len(rf.indices), n_estimators)

        it_checker = zip(
            rf.estimators,
            rf.indices,
            object_sampler.indices,
            feature_sampler.indices,
        )

        for estimator, indices, objects_index, features_index in it_checker:
            objects_index = np.asarray(objects_index)
            features_index = np.asarray(features_index)

            self.assertTrue(np.all(indices == features_index))
            self.assertTrue(np.all(estimator.X == x[np.ix_(objects_index, features_index)]))
            self.assertTrue(np.all(estimator.y == y[objects_index]))

    def test_check_predicts(self):
        object_sampler = PredefinedSampler([])

        feature_sampler = PredefinedSampler([
            [2, 3, 3],
            [0, 1, 2],
            [3, 2, 1],
            [3, 0, 1],
        ])

        n_estimators = len(feature_sampler.indices)
        n_features = max(chain.from_iterable(feature_sampler.indices), default=-1) + 1

        rf = RandomForest(
            base_estimator=RandomClassifier,
            object_sampler=object_sampler,
            feature_sampler=feature_sampler,
            n_estimators=n_estimators,
        )

        x = np.asarray([[1] * n_features] * n_estimators)
        y = np.asarray([0] * n_estimators)

        estimators = []
        for i in range(n_estimators):
            estimator = RandomClassifier(n_classes=3, random_state=1000 + i)
            estimator = estimator.fit(x, y)
            estimators.append(estimator)

        rf.estimators = estimators.copy()
        rf.indices = feature_sampler.indices

        x = np.arange(10 * n_features).reshape(10, n_features)

        y_predict = rf.predict_proba(x)
        y_expected = np.asarray([
            [0.42165396, 0.31929012, 0.39484171],
            [0.52803114, 0.49340856, 0.34016372],
            [0.13338973, 0.54480853, 0.60197650],
            [0.43311089, 0.53526597, 0.44281023],
            [0.40348041, 0.56022464, 0.29619213],
            [0.31099268, 0.46019327, 0.35188282],
            [0.62788872, 0.20714953, 0.47497037],
            [0.70099521, 0.34210022, 0.44912750],
            [0.48720505, 0.41713190, 0.52659920],
            [0.59248042, 0.29705678, 0.81229349],
        ])

        self.assertTrue(np.allclose(y_expected, y_predict))
