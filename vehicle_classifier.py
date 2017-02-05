#-*- coding: utf-8 -*-

"""
"""

import time

import glob

import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import config


class Dataset(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y


class ClassificationData(object):

    def __init__(self):
        self.training_data = None
        self.test_data = None
        self.validation_data = None

    def stats(self):
        """collect stats for classification data"""

        d = {
            'n_training_samples': len(self.training_data.X)
        }
        if self.test_data:
            d['n_test_samples'] = len(self.test_data.X)
        if self.validation_data:
            d['n_validation_samples'] = len(self.validation_data.X)
        return d


class VehicleClassifier(object):

    def __init__(self):
        self.model = None

    def build_model(self):
        """builds a classifier model"""

        self.model = LinearSVC()

    def save_model(self):
        """pickles the model to file system"""

        joblib.dump(self.model, config.model_path)

    def load_model(self):
        """loads an already trained model"""

        self.model = joblib.load(config.model_path)

    def train(self, data):
        """trains the model"""

        t0 = time.time()
        self.model.fit(data.X, data.y)
        t1 = time.time()

        print("{:.2f} seconds to train the model".format(t1-t0))

    def test(self, data):
        """tests the model on test set"""

        t0 = time.time()
        score = self.model.score(data.X, data.y)
        t1 = time.time()

        print("{:.2f} seconds to test the model, accuracy: {:.4f}".format(
            t1-t0, score
        ))

    def classify(self, X):
        """makes prediction using the trained model"""

        return self.model.predict(X)

    def collect_data(self, feature_extractor):
        """collects data and splits it up on training/test/cross-val set"""

        car_files = self.get_image_files(config.vehicle_data)
        car_features = feature_extractor.extract_features(car_files)
        print("n_car_samples: {}".format(len(car_features)))

        not_car_files = self.get_image_files(config.non_vehicles_data)
        not_car_features = feature_extractor.extract_features(not_car_files)
        print("n_not_car_samples: {}".format(len(not_car_features)))

        print("n_features: {}".format(len(car_features[0])))

        # Create an array stack of feature vectors
        X = np.vstack((car_features, not_car_features)).astype(np.float64)

        # Normalize the data
        X_scaler = StandardScaler().fit(X)
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack(
            (np.ones(len(car_features)), np.zeros(len(not_car_features)))
        )

        data = ClassificationData()

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y,
            test_size=config.test_train_ratio,
            random_state=rand_state
        )
        data.training_data = Dataset(X_train, y_train)

        if not config.validation_test_ratio:
            data.test_data = Dataset(X_test, y_test)
        else:
            X_val, X_test, y_val, y_test = train_test_split(
                X_test, y_test,
                test_size=config.validation_test_ratio,
                random_state=rand_state
            )
            data.validation_data = Dataset(X_val, y_val)
            data.test_data = Dataset(X_test, y_test)

        return data

    @classmethod
    def get_image_files(cls, base_path):
        files = []
        for fmt in ['.png', '.jpeg', '.jpg']:
            files.extend(glob.glob(
                '{}/**/*{}'.format(base_path, fmt),
                recursive=True
            ))
        return files

    @classmethod
    def get_vehicle_classifier(cls):
        # TODO:
        pass


def main():
    from feature_extractor import FeatureExtractor
    feature_extractor = FeatureExtractor.get_feature_extractor()
    classifier = VehicleClassifier()
    data = classifier.collect_data(feature_extractor)
    classifier.build_model()
    classifier.train(data.training_data)
    classifier.test(data.test_data)
    classifier.save_model()


if __name__ == "__main__":
    main()
