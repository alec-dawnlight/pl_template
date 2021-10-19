from .base import BaseLitModel
from .cnn_lit import CNNLitModel

from .metrics import *

def get_model_class():
    '''
    Implement logic here for choosing how to get the correct lit model.
    Could be by loss, metrics, training/eval procedure, etc.
    '''
    return CNNLitModel
