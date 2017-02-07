#-*- coding: utf-8 -*-

"""
"""

import glob
import pickle
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config
from feature_extractor import FeatureExtractor


class Dataset(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y


class ClassificationData(object):

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.validation_data = None

    def stats(self):
        """collect stats for classification data"""

        d = {
            'n_training_samples': len(self.train_data.X)
        }
        if self.test_data:
            d['n_test_samples'] = len(self.test_data.X)
        if self.validation_data:
            d['n_validation_samples'] = len(self.validation_data.X)
        return d


class VehicleClassifier(object):

    def __init__(self, feature_extractor=None):
        self.model = None
        self.scaler = None
        if feature_extractor:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = FeatureExtractor.get_feature_extractor()

    def build_model(self):
        """builds a classifier model"""

        self.model = LinearSVC()
        self.scaler = StandardScaler()

    def save(self, fname=None):
        """pickles the model to file system"""

        fname = fname or config.model_path
        with open(fname, 'wb') as f:
            pickle.dump(
                {
                    'classifier': self.model,
                    'scaler': self.scaler
                },
                f
            )

    def load(self, fname=None):
        """loads an already trained model"""

        fname = fname or config.model_path
        with open(fname, 'rb') as f:
            d = pickle.load(f)
            self.model = d['classifier']
            self.scaler = d['scaler']

    def train(self, data):
        """trains the model"""

        t0 = time.time()
        self.model.fit(data.train_data.X, data.train_data.y)
        t1 = time.time()

        print("{:.2f} seconds to train the model".format(t1-t0))

        if data.test_data:
            t0 = time.time()
            score = self.model.score(data.test_data.X, data.test_data.y)
            t1 = time.time()

            print("{:.2f} seconds to test the model, accuracy: {:.4f}".format(
                t1 - t0, score
            ))

    # def test(self, data):
    #     """tests the model on test set"""
    #
    #     t0 = time.time()
    #     score = self.model.score(data.X, data.y)
    #     t1 = time.time()
    #
    #     print("{:.2f} seconds to test the model, accuracy: {:.4f}".format(
    #         t1-t0, score
    #     ))

    def classify(self, X):
        """makes prediction using the trained model"""

        return self.model.predict(X)

    def prepare_training_data(self, vehicles_data=None, non_vehicles_data=None):
        """collects data and splits it up on training/test/cross-val set"""

        car_files = self.get_image_files(vehicles_data or config.vehicles_data)
        car_features = self.feature_extractor.extract_features(car_files)
        print("n_car_samples: {}".format(len(car_features)))

        not_car_files = self.get_image_files(
            non_vehicles_data or config.non_vehicles_data)
        not_car_features = self.feature_extractor.extract_features(
            not_car_files)
        print("n_not_car_samples: {}".format(len(not_car_features)))

        print("n_features: {}".format(len(car_features[0])))

        # Create an array stack of feature vectors
        X = np.vstack((car_features, not_car_features)).astype(np.float64)

        # Normalize the data
        self.scaler = self.scaler.fit(X)
        scaled_X = self.scaler.transform(X)

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
        data.train_data = Dataset(X_train, y_train)

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

        print("classification data stats: {}".format(data.stats()))

        return data

    def predict(self, image_list):
        """makes prediction for the given list of images/files"""

        features = self.feature_extractor.extract_features(image_list)
        print('n_samples: {}'.format(len(features)))
        print('n_features: {}'.format(len(features[0])))

        X = np.vstack((features, )).astype(np.float64)
        scaled_X = self.scaler.transform(X)

        t0 = time.time()
        prediction = self.model.predict(scaled_X)
        t1 = time.time()

        print('prediction: {} in {:.2f} seconds'.format(prediction, t1-t0))

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
    feature_extractor = FeatureExtractor.get_feature_extractor()
    classifier = VehicleClassifier(feature_extractor)
    classifier.build_model()
    data = classifier.prepare_training_data()
    classifier.train(data)
    classifier.save()


if __name__ == "__main__":
    main()
