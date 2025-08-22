import torch
import torch.nn as nn
from torch.autograd import Variable
from config import device
from data import NO_LABEL

def validate(eval_loader, model):
    model.eval() # set model to evaluation mode
    total = 0
    correct = 0

    for i, (input,target) in enumerate(eval_loader):

        with torch.no_grad():
            input_var = Variable(input).to(device) # move input to device (CPU)
            target_var = Variable(target).to(device) # move target to device (CPU) 
        
        labeled_minibatch_size = target_var.ne(NO_LABEL).sum() # number of labeled samples in the minibatch

        assert labeled_minibatch_size > 0, "No labeled samples in the minibatch"
        output = model(input_var) # forward pass

        _ , predicted = torch.max(output.data, 1) # get the index of the max log-probability
        total += target_var.size(0) 
        correct += (predicted == target_var).sum().item() # count of correct predictions
    
    return 100 * correct / total