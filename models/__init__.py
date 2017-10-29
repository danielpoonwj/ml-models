from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os

# temporary for development
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
db_path = os.path.join(CURRENT_DIR, 'mock.db')

engine = create_engine('sqlite:////%s' % db_path)

Session = sessionmaker()
Session.configure(bind=engine)

Base = declarative_base()
