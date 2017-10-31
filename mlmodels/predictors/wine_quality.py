from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from mlmodels.predictors.base import BasePredictor


class WineQuality(BasePredictor):
    def __init__(self, version=1):
        BasePredictor.__init__(self, version=version)

        # ignore 'index' when getting data
        self.add_ignore_column('index')

    def train(self):
        data = self.get_data()

        y = data['quality']
        X = data.drop('quality', axis=1)

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
