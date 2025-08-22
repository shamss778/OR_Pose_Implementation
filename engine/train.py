
import config
import Mean_Teacher_Method.utils.ramps as ramps
import torch
import torch.nn as nn
from torch.autograd import Variable
import Mean_Teacher_Method.engine.metrics as metrics


from data import NO_LABEL
import state
from Mean_Teacher_Method.utils.loss import softmax_mse_loss, symmetric_mse_loss
from engine.hooks import update_ema_variables, adjust_learning_rate, get_current_consistency_weight

device = torch.device("cpu")

def train(train_loader, model, ema_model, optimizer, epoch):

    # average meters to record the training statistics
    running_loss = 0.0     

    # Loss function of student with reduction 'sum' to devide by the total number of labeled samples
    # in the minibatch instead of the total batch size 
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL)     

    losses = metrics.AverageMeter() # to record the average loss

    # Loss function of consistency loss size between student and teacher
    consistency_criterion = softmax_mse_loss  

    # Loss function of the distance between two logits of the student in multi-head architecture
    residual_logit_criterion = symmetric_mse_loss 

    # switch to train mode
    model.train()
    ema_model.train()

    # Iterate over the training data for one epoch
    # train loader provides both labeled and unlabeled data
    for i, (input, target) in enumerate(train_loader): 

        # skip the last incomplete batch
        if input.size(0) != config.batch_size:
            continue

        # Adjust the learning rate 
        ramps.adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # Move input and target to device (CPU)
        input_var = Variable(input).to(device)
        target_var = Variable(target).to(device)
        ema_input_var = Variable(input).to(device)

        minibatch_size = input.size(0)
        labeled_minibatch_size = target_var.ne(NO_LABEL).sum()   

        # Make sure we have at least one labeled sample in the minibatch
        assert labeled_minibatch_size > 0, "No labeled samples in the minibatch"

        # Compute output of student and teacher
        student_out = model(input_var)
        teacher_out = ema_model(ema_input_var)            

        # if semi-supervised learning
        if not config.supervised:
            
            # Multi-head architecture
            if isinstance(student_out, (tuple, list)):
                assert len(student_out) == 2 and len(teacher_out) == 2, "For multi-head architecture, "
                "the model should return two outputs"
                student_logit1, student_logit2 = student_out # two heads of student
                teacher_logit, _ = teacher_out
    
            else:
                assert config.logit_distance_cost <= 0, "For multi-head architecture, set logit_distance_cost > 0"
                student_logit = student_out
                teacher_logit = teacher_out
        
            teacher_logit = teacher_logit.detach()  # we do not need to backpropagate through teacher

            # compute losses with or without multi-head architecture
            if config.logit_distance_cost >= 0:
                # logit1 is for classification, logit2 is for consistency
                classification_logit, consistency_logit = student_logit1, student_logit2
                residual_loss = config.logit_distance_cost * residual_logit_criterion(classification_logit, consistency_logit) / minibatch_size 
            else:
                classification_logit, consistency_logit = student_logit, student_logit
                residual_loss = 0  

            # Compute classification loss of the labeled samples + unlabeled (only for validation)
            classification_loss = class_criterion(classification_logit, target_var) / labeled_minibatch_size
            ema_classification_loss = class_criterion(teacher_logit, target_var) / labeled_minibatch_size # for monitoring only
            
            if config.consistency:
                # Compute consistency loss of all samples
                consistency_weight = get_current_consistency_weight(epoch)
                consistency_loss = consistency_weight * consistency_criterion(consistency_logit, teacher_logit) / minibatch_size

            # Total loss
            loss = classification_loss + consistency_loss + residual_loss

        else: # Supervised learning
            classification_loss = class_criterion(student_out, target_var) / labeled_minibatch_size
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

        # print training stats every 50 mini-batches
        if i % 50 == 49:    
            if not config.supervised:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss:.4f}\t'
                      'Class Loss {class_loss:.4f}\t'
                      'Ema Class Loss {ema_class_loss:.4f}\t'
                      'Consistency Loss {consistency_loss:.4f}\t'
                      'Residual Loss {residual_loss:.4f}\t'.format(
                       epoch, i + 1, len(train_loader), loss=running_loss / 50,
                       class_loss=classification_loss.item(),
                       ema_class_loss=ema_classification_loss.item(),
                       consistency_loss=consistency_loss.item() if config.consistency else 0,
                       residual_loss=residual_loss.item() if config.logit_distance_cost >= 0 else 0))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss:.4f}\t'
                      'Class Loss {class_loss:.4f}\t'.format(
                       epoch, i + 1, len(train_loader), loss=running_loss / 50,
                       class_loss=classification_loss.item()))
            running_loss = 0.0

    return losses, running_loss






