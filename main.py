from config import device
from state import global_step, best_prec1
from data.dataLoader import train_dataset, test_dataset, train_loader, test_loader
from engine import train, validate
import torch
from data.data import relabel_dataset
from data.dataLoader import test_loader
from models import curiousAI_model
import torch.backends.cudnn as cudnn
import utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
import os
from engine.validate import validate
from engine.train import train
from torch.utils.tensorboard import SummaryWriter

def main(config):
    global global_step, best_prec1
    
    # Initialize student and teacher models
    student_model = curiousAI_model.MODEL_REGISTRY[config.model](config, data=None).to(device) 
    teacher_model = curiousAI_model.MODEL_REGISTRY[config.model](config,nograd= True, data=None).to(device)  
    
    optimizer = torch.optim.SGD(student_model.parameters(), 
                                lr=config.lr, 
                                momentum=config.momentum, 
                                weight_decay=config.weight_decay)   
    
    cudnn.benchmark = True # For fast training

    # Prepare save directory
    save_path, _ = utils.checkpoint.prepare_save_dir(config)

    # Optionally resume from a checkpoint
    """         assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))"""

    # TensorBoard writer
    test_writer = SummaryWriter(os.path.join(save_path, 'test')) 

    # Evaluation before training for sanity check
    if config.evaluate:
        print("Evaluating the student model:")
        acc1 = validate(eval_loader=test_loader, model=student_model)
        print(f"Student Model Accuracy: {acc1:.2f}%")
        print("Evaluating the teacher model:")
        acc2 = validate(eval_loader=test_loader, model=teacher_model)
        print(f"Teacher Model Accuracy: {acc2:.2f}%")

    for epoch in range(config.start_epoch, config.epochs):
        
        """ Train for one epoch """

        # Train function in engine.py already ready
        train(train_loader, student_model, teacher_model, optimizer, epoch)

        """ Evaluate on validation set """
        if config.evaluation_epochs and (epoch + 1) % config.evaluation_epochs == 0:
            
            prec1 = validate(test_loader, student_model)
            ema_prec1 = validate(test_loader, teacher_model)

            print('Accuracy of the Student network on the 10000 test images: %d %%' % (
                prec1))
            print('Accuracy of the Teacher network on the 10000 test images: %d %%' % (
                ema_prec1))
            
            test_writer.add_scalar('Accuracy Student', prec1, epoch)
            test_writer.add_scalar('Accuracy Teacher', ema_prec1, epoch)

            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if config.checkpoint_epochs and (epoch + 1) % config.checkpoint_epochs == 0:
            utils.checkpoint.save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': config.model,
                'state_dict': student_model.state_dict(),
                'ema_state_dict': teacher_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path, epoch + 1)