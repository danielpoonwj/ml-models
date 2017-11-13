import os

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from mlmodels.predictors.base import BasePredictor
from mlmodels.utils import time_taken

# reference mock sqlite db
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
db_path = os.path.join(os.path.dirname(CURRENT_DIR), 'models', 'mock.db')
engine = create_engine('sqlite:////%s' % db_path)


class WineQuality(BasePredictor):
    def __init__(self, version=1):
        super().__init__('quality', version=version)

        # ignore 'index' when getting data
        self.add_ignore_column('index')

    @time_taken
    def train(self):
        print('Training started')

        data = self.get_data(engine)

        y = data[self.prediction_field]
        X = data.drop(self.prediction_field, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

        pipeline = make_pipeline(
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100)
        )

        hyperparameters = {
            'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
            'randomforestregressor__max_depth': [None, 5, 3, 1]
        }

        clf = GridSearchCV(pipeline, hyperparameters, cv=10)
        clf.fit(X_train, y_train)

        self.clf = clf
