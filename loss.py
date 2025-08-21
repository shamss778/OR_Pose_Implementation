from torch.nn import functional as F
import torch

def softmax_mse_loss(input_logit, target_logit): 
    """Consistency loss function that computes the mean squared error"""
    assert input_logit.size() == target_logit.size(), "Input and target logits must have the same size"
    # Apply activation function (softmax) to the logits
    # to convert them into probabilities
    input_softmax = F.softmax(input_logit, dim=1) 
    target_softmax = F.softmax(target_logit, dim=1)
    num_classes = input_logit.size()[1] # Number of classes
    # Compute the mean squared error loss between the softmax outputs
    # Normalize by the number of classes
    loss = F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes
    return loss

def symmetric_mse_loss(input1, input2):
    """going to use this function for computing residual loss"""
    assert input1.size() == input2.size(), "Input tensors must have the same size"
    num_classes = input1.size()[1] # Number of classes
    # Compute the mean squared error loss between the two inputs
    loss = F.mse_loss(input1, input2, reduction='sum') / num_classes
    return loss     

