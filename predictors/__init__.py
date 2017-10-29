import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'output')

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
