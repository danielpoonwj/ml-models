import os
from datetime import datetime

from io import BytesIO
import pandas as pd
from sqlalchemy.orm import defer, class_mapper

from sklearn.base import BaseEstimator
from sklearn.externals import joblib

import boto3

from mlmodels.models import Session, db


def camel_to_snake(input_text):
    temp_list = []
    for idx, c in enumerate(input_text):
        if c.isupper() and idx > 0:
            temp_list.append('_')
        temp_list.append(c)
    return ''.join(temp_list).lower()


class BasePredictor:
    def __init__(self, prediction_field, version=1):
        self.predictor_name = camel_to_snake(self.__class__.__name__)
        self.version = version
        self.bucket = os.environ['AWS_BUCKET_NAME']

        self.prediction_field = prediction_field
        self.ignore_columns = set()

        self.model = getattr(db, '%sModel' % self.__class__.__name__)

        self.clf = None
        self.clf_version = None

        self.s3_prefix = '{predictor_name}/v{version}'.format(
            predictor_name=self.predictor_name,
            version=self.version
        )

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

        file_buffer = BytesIO()
        joblib.dump(self.clf, file_buffer)
        file_buffer.seek(0)

        s3 = boto3.resource('s3')

        self.clf_version = '{s3_prefix}/{timestamp}'.format(
            s3_prefix=self.s3_prefix,
            timestamp=int(datetime.timestamp(datetime.now()))
        )

        key_name = '%s.pkl' % self.clf_version

        s3.Bucket(self.bucket).put_object(
            Key=key_name,
            Body=file_buffer
        )

    def get_latest_key(self, to_underscore=False):
        s3 = boto3.client('s3')

        object_list = s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.s3_prefix
        ).get('Contents', [])

        object_list.sort(key=lambda x: x.get('Key'))
        object_key = object_list.pop()['Key']

        if to_underscore:
            object_key = object_key.replace('/', '_')

        return object_key

    def load(self):
        s3 = boto3.client('s3')

        object_key = self.get_latest_key()
        self.clf_version = object_key.replace('.pkl', '')

        resp = s3.get_object(
            Bucket=self.bucket,
            Key=object_key
        )

        pickle_buffer = BytesIO(resp['Body'].read())
        self.clf = joblib.load(pickle_buffer)

    def predict(self, entry):
        assert isinstance(entry, dict)
        entry = pd.DataFrame([entry])

        return self.clf.predict(entry)[0]
