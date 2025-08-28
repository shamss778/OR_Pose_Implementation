
import config
import utils.ramps as ramps
import torch
import torch.nn as nn
from torch.autograd import Variable
import engine.metrics as metrics


from data.data import NO_LABEL
import state
from utils.loss import softmax_mse_loss, symmetric_mse_loss
from engine.hooks import update_ema_variables, adjust_learning_rate, get_current_consistency_weight

device = torch.device("cpu")

def train(train_loader, model, ema_model, optimizer, epoch):

    # average meters to record the training statistics
    running_loss = 0.0     
    losses = metrics.AverageMeter() # to record the average loss
    
    # Loss function of student with reduction 'sum' to devide by the total number of labeled samples
    # in the minibatch instead of the total batch size 
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL)     

    # Loss function of consistency loss size between student and teacher
    consistency_criterion = softmax_mse_loss  

    # switch to train mode
    model.train()
    ema_model.train()

    # Iterate over the training data for one epoch
    # train loader provides both labeled and unlabeled data
    for i, ((input, ema_input), target) in enumerate(train_loader): 

        # skip the last incomplete batch
        if input.size(0) != config.batch_size:
            continue

        # Adjust the learning rate 
        ramps.adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # Move input and target to device (CPU)
        input_var = Variable(input).to(device)
        target_var = Variable(target).to(device)

        minibatch_size = len(target_var)  # total minibatch size (labeled + unlabeled)
        labeled_minibatch_size = target_var.ne(NO_LABEL).sum()   

        # Make sure we have at least one labeled sample in the minibatch
        assert labeled_minibatch_size > 0, "No labeled samples in the minibatch"

        # Compute output of student and teacher
        student_out = model(input_var)  

        classification_loss = class_criterion(student_out, target_var) / minibatch_size        

        # if semi-supervised learning
        if not config.supervised:

            # compute output of teacher
            with torch.no_grad():
                ema_input_var = Variable(ema_input).to(device)

            ema_model_out = ema_model(ema_input_var)

            ema_logit = Variable(ema_model_out.detach().data, requires_grad=False).to(device)

            if config.consistency:
                # Compute consistency loss of all samples
                consistency_weight = get_current_consistency_weight(epoch)
                consistency_loss = consistency_weight * consistency_criterion(student_out, ema_logit) / minibatch_size
            else:
                consistency_loss = 0
            
            # Total loss
            loss = classification_loss + consistency_loss

        else: # Supervised learning
            loss = classification_loss

        # compute gradient and do SGD step
        optimizer.zero_grad() # clear previous gradients
        loss.backward()      # backpropagate the loss
        optimizer.step()    # update the parameters
        state.global_step = state.global_step + 1 # increment global step which is used for EMA update
        
        if not config.supervised:
            # update the teacher with exponential moving average of the student
            update_ema_variables(model, ema_model, config.ema_decay, state.global_step)

        # print statistics
        running_loss += loss.item()

        if i % 20 == 19:    # print every 20 mini-batches
            print('[Epoch: %d, Iteration: %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

        losses.update(loss.item(), input.size(0))

    return losses, running_loss






