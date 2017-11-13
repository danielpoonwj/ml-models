import os
from datetime import datetime

from io import BytesIO
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import defer, class_mapper

from sklearn.base import BaseEstimator
from sklearn.externals import joblib

import boto3

from mlmodels.models import db
from mlmodels.utils import camel_to_snake


class BasePredictor:
    def __init__(self, prediction_field, version=1):
        self.predictor_name = camel_to_snake(self.__class__.__name__)
        self.version = version

        self.prediction_field = prediction_field
        self.ignore_columns = set()

        self.engine = None
        self.model = getattr(db, '%sModel' % self.__class__.__name__)

        self.clf = None
        self.clf_version = None

        self.folder_prefix = '{predictor_name}/v{version}'.format(
            predictor_name=self.predictor_name,
            version=self.version
        )

        self._save_mode = os.environ.get('ML_MODELS_MODE', 'local')
        assert self._save_mode in ('local', 'aws')

        # validation for necessary aws environment variables
        if self._save_mode == 'aws':
            for env_var_name in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_BUCKET_NAME'):
                env_var_val = os.environ.get(env_var_name)
                assert env_var_val is not None and len(env_var_val) > 0, \
                    '%s has to be set if mode=aws' % env_var_name

        # unused if save mode is local
        self.bucket = os.environ.get('AWS_BUCKET_NAME')

    def add_ignore_column(self, column_name):
        self.ignore_columns.add(column_name)

    def get_column_types(self):
        return {
            column.name: column.type.python_type
            for column in class_mapper(self.model).columns
            if column.name not in self.ignore_columns
        }

    def get_data(self):
        assert self.engine is not None

        Session = sessionmaker()
        Session.configure(bind=self.engine)

        session = Session()
        try:
            query = session \
                .query(self.model) \
                .options(*[defer(col) for col in self.ignore_columns])

            data = pd.read_sql(query.statement, query.session.bind)
        finally:
            session.close()

        return data

    def _local_dir(self):
        models_dir = os.environ.get('ML_MODELS_DIR', '/tmp/ml_models')
        write_dir = os.path.join(models_dir, self.folder_prefix)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        return write_dir

    def save(self):
        getattr(self, '_save_%s' % self._save_mode)()

    def _save_local(self):
        assert isinstance(self.clf, BaseEstimator)
        current_timestamp = int(datetime.timestamp(datetime.now()))

        write_path = os.path.join(self._local_dir(), '%s.pkl' % current_timestamp)
        joblib.dump(self.clf, write_path)

        self.clf_version = '{folder_prefix}/{timestamp}'.format(
            folder_prefix=self.folder_prefix,
            timestamp=current_timestamp
        )

        print('Model saved to local path: %s' % write_path)

    def _save_aws(self):
        assert isinstance(self.clf, BaseEstimator)

        file_buffer = BytesIO()
        joblib.dump(self.clf, file_buffer)
        file_buffer.seek(0)

        s3 = boto3.resource('s3')

        self.clf_version = '{folder_prefix}/{timestamp}'.format(
            folder_prefix=self.folder_prefix,
            timestamp=int(datetime.timestamp(datetime.now()))
        )

        key_name = '%s.pkl' % self.clf_version

        s3.Bucket(self.bucket).put_object(
            Key=key_name,
            Body=file_buffer
        )

        print('Model saved to S3 object: %s' % key_name)

    def get_latest(self):
        return getattr(self, '_get_latest_%s' % self._save_mode)()

    def _get_latest_local(self):
        file_names = os.listdir(self._local_dir())
        file_names.sort()
        
        return file_names.pop()

    def _get_latest_aws(self):
        s3 = boto3.client('s3')

        object_list = s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.folder_prefix
        ).get('Contents', [])

        object_list.sort(key=lambda x: x.get('Key'))
        object_key = object_list.pop()['Key']

        return object_key

    def load(self):
        getattr(self, '_load_%s' % self._save_mode)()
    
    def _load_local(self):
        latest_file = self._get_latest_local()

        read_path = os.path.join(self._local_dir(), latest_file)
        self.clf = joblib.load(read_path)
        self.clf_version = '%s/%s' % (self.folder_prefix, latest_file.replace('.pkl', ''))

        print('Model loaded from local path: %s' % read_path)
    
    def _load_aws(self):
        s3 = boto3.client('s3')

        object_key = self._get_latest_aws()

        resp = s3.get_object(
            Bucket=self.bucket,
            Key=object_key
        )

        pickle_buffer = BytesIO(resp['Body'].read())
        self.clf = joblib.load(pickle_buffer)
        self.clf_version = object_key.replace('.pkl', '')

        print('Model loaded from S3 object: %s' % object_key)

    def predict(self, entry):
        assert isinstance(entry, dict)
        entry = pd.DataFrame([entry])

        return self.clf.predict(entry)[0]
