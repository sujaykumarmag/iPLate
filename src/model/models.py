from src.model.cnn import CNN
import torch 
import torch.nn as nn

model = CNN(215)

models = []
model1 = {"name":"CNN_Baseline","model":model}
models.append(model1)

def get_models():
    return models
