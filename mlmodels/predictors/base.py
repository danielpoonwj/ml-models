import os
from datetime import datetime

import pandas as pd
from sqlalchemy.orm import defer, class_mapper

from sklearn.base import BaseEstimator
from sklearn.externals import joblib

import boto3

from mlmodels.models import Session, db

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'tmp')

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def camel_to_snake(input_text):
    temp_list = []
    for idx, c in enumerate(input_text):
        if c.isupper() and idx > 0:
            temp_list.append('_')
        temp_list.append(c)
    return ''.join(temp_list).lower()


class BasePredictor:
    def __init__(self, version=1):
        self.predictor_name = camel_to_snake(self.__class__.__name__)
        self.version = version

        self.ignore_columns = set()

        self.model = getattr(db, '%sModel' % self.__class__.__name__)

        self.clf = None

    def add_ignore_column(self, column_name):
        self.ignore_columns.add(column_name)

    def get_column_types(self):
        return {
            column.name: column.type.python_type
            for column in class_mapper(self.model).columns
            if column.name not in self.ignore_columns
        }

    def get_data(self):
        session = Session()
        try:
            query = session \
                .query(self.model) \
                .options(*[defer(col) for col in self.ignore_columns])

            data = pd.read_sql(query.statement, query.session.bind)
        finally:
            session.close()

        return data

    def save(self):
        """
        Save trained timestamped model
        """

        assert isinstance(self.clf, BaseEstimator)

        current_timestamp = int(datetime.timestamp(datetime.now()))

        file_name = '{predictor_name}_v{version}_{timestamp}.pkl'.format(
            predictor_name=self.predictor_name,
            version=self.version,
            timestamp=current_timestamp
        )

        write_path = os.path.join(OUTPUT_DIR, file_name)

        joblib.dump(self.clf, write_path)

        bucket_name = os.environ['AWS_BUCKET_NAME']

        s3 = boto3.client('s3')

        key_name = '{predictor_name}/v{version}/{timestamp}.pkl'.format(
            predictor_name=self.predictor_name,
            version=self.version,
            timestamp=current_timestamp
        )

        s3.upload_file(
            write_path,
            bucket_name,
            key_name
        )

        os.remove(write_path)
