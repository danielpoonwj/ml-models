import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import defer

from sklearn.base import BaseEstimator
from sklearn.externals import joblib

from models import Session
import models.db

from predictors import OUTPUT_DIR


class BasePredictor:
    def __init__(self, predictor_name, version=1):
        self.predictor_name = predictor_name
        self.version = version

        self.clf = None

    def get_data(self, *ignore_columns):
        model_name = '%sModel' % self.__class__.__name__
        db_model = getattr(models.db, model_name)

        session = Session()
        try:
            query = session \
                .query(db_model) \
                .options(*[defer(col) for col in ignore_columns])

            data = pd.read_sql(query.statement, query.session.bind)
        finally:
            session.close()

        return data

    def save(self):
        """
        Save trained timestamped model
        """

        assert isinstance(self.clf, BaseEstimator)
        file_name = '{predictor_name}_v{version}_{timestamp}.pkl'.format(
            predictor_name=self.predictor_name,
            version=self.version,
            timestamp=int(datetime.timestamp(datetime.now()))
        )

        write_path = os.path.join(OUTPUT_DIR, file_name)

        joblib.dump(self.clf, write_path)
