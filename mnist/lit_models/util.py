'''
Implement any utilities that are not related to a specific model here
'''
import torch.nn.functional as F

def to_onehot(t, num_classes):
    return F.one_hot(t, num_classes)