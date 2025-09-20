import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from config import device, build_config
import state
import torch
from data.data import TransformTwice, GaussianNoise, relabel_dataset, TwoStreamBatchSampler as TransfromTwice, GaussianNoise, relabel_dataset, TwoStreamBatchSampler
from models import curiousAI_model, EncoderDecoder
import torch.backends.cudnn as cudnn
import utils.checkpoint
from torch.utils.tensorboard import SummaryWriter
from engine.validate import validate
from engine.train import train
from torch.utils.tensorboard import SummaryWriter
import config
import torchvision
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader
import models
import data.data as data


def main(config):


    train_transform = TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,  0.2435,  0.2616))]))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470,  0.2435,  0.2616))
    ])

    train_dataset = torchvision.datasets.ImageFolder(config.traindir, train_transform)
    test_dataset = torchvision.datasets.ImageFolder(config.testdir, test_transform)

    if config.labels:
        with open(config.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = relabel_dataset(train_dataset, labels)

    if config.labeled_batch_size:# if using semi-supervised learning
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, config.batch_size, config.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(config.labeled_batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler= batch_sampler,
        num_workers= config.num_workers,
        pin_memory= config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size= config.batch_size,
        shuffle=False,
        num_workers= 2 * config.num_workers,
        pin_memory= config.pin_memory,
        drop_last=False
    )
    
    def create_model(num_classes, ema=False):
        
 
        if config.model not in models.curiousAI_model.__dict__:
            raise ValueError("Invalid model architecture: {}".format(config.model))

        model_factory = models.curiousAI_model.__dict__[config.model]
        model_params = dict(pretrained=config.pretrained, num_classes=num_classes)
        model = model_factory(**model_params)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model
    

    # Initialize student and teacher models
    student_model = create_model(num_classes=10, ema=False).to(device)
    teacher_model = create_model(num_classes=10, ema=True).to(device)
    
    optimizer = torch.optim.SGD(student_model.parameters(), 
                                lr=config.lr, 
                                momentum=config.momentum, 
                                weight_decay=config.weight_decay)   
    
    cudnn.benchmark = True # For fast training

    # Prepare save directory
    # save_path, _ = utils.checkpoint.prepare_save_dir(config)

    # Optionally resume from a checkpoint
    """         assert os.path.isfile(config.resume), "=> no checkpoint found at '{}'".format(config.resume)
        LOG.info("=> loading checkpoint '{}'".format(config.resume))
        checkpoint = torch.load(config.resume)
        config.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(config.resume, checkpoint['epoch']))"""

    # TensorBoard writer
    ''' test_writer = SummaryWriter(os.path.join(save_path, 'test')) '''

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

            is_best = ema_prec1 > state.best_prec1
            state.best_prec1 = max(ema_prec1, state.best_prec1)
        else:
            is_best = False

'''        if config.checkpoint_epochs and (epoch + 1) % config.checkpoint_epochs == 0:
            utils.checkpoint.save_checkpoint({
                'epoch': epoch + 1,
                'global_step': state.global_step,
                'arch': config.model,
                'state_dict': student_model.state_dict(),
                'ema_state_dict': teacher_model.state_dict(),
                'best_prec1': state.best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, save_path, epoch + 1)'''

if __name__ == "__main__":
    config = build_config()
    print(models.curiousAI_model.__dict__.keys())
    main(config)
