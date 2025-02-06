import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Model')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Result')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)