import torch
import onnx
from onnx import numpy_helper
import copy

def onnx_to_torch(onnx_model, torch_model, save_to : str = None): 
    '''
    Initializes the parameters of a torch model from its onnx counterpart.
    Provide a filepath to save the initialized torch model. 
    '''
    graph = onnx_model.graph
    initializers = dict()

    for init in graph.initializer: 
        initializers[init.name] = numpy_helper.to_array(init)

    torch_model = copy.deepcopy(torch_model)

    for name, p in torch_model.named_parameters(): 
        p.data = (torch.from_numpy(initializers[name])).data

    if save_to is not None: 
        torch_model.save(save_to)

    return torch_model
