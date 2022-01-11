import os
import torch
import torch.nn as nn
from Models.models import SegmentationModule, build_encoder, build_decoder
from Models.dataset import TrainDataset


def pixel_acc(pred, label):
    """
        Function for calculating the pixel accuracy between the predicted image and labeled image
    """
    _, preds = torch.max(pred, dim=1)
    valid = (label>=0).long() # some labels are -1 and are ignored
    acc_sum = torch.sum(valid * (preds==label).long())
    pixel_sum = torch.sum(valid)
    return acc_sum.float() / (pixel_sum.float() + 1e-10)


def train_one_epoch(segmentation_module, iterator, optimizers, history, epoch, crit, start_lr=0.02):
    """
        Function for training model for only one epoch
    """
    
    # initialize helper variables for calucating average loss and accuracy in one epoch
    total_loss = 0
    total_acc = 0
    segmentation_module.train()
    
    # Training constants
    total_num_iter = 1e5
    num_iter_per_epoch = 5000
     
    start_learn_rate = start_lr
    
    for i in range(num_iter_per_epoch):
        # load a batch of data
        batch_data = next(iterator)[0]
        segmentation_module.zero_grad() 
        
        # adjust learning rate (learning rate "poly")
        cur_iter = i + (epoch - 1) * num_iter_per_epoch
        adjust_learning_rate(optimizers, cur_iter, start_learn_rate, total_num_iter)
                
        # forward pass
        pred = segmentation_module(batch_data)
        
        # Calculate loss and accuracy
        loss = crit(pred, batch_data['seg_label'].to('cuda'))
        acc  = pixel_acc(pred, batch_data['seg_label'].to('cuda'))
               
        loss = loss.mean()
        acc = acc.mean()

        # Backward pass
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # update average loss and acc
        total_loss += loss.data.item()
        total_acc += acc.data.item()*100

        # Displaying epoch, iteration, average accuracy and average loss every 20 iterations
        if i%20 == 0 and i>0:            
            ave_total_loss = total_loss / (i+1)
            ave_acc = total_acc / (i+1)
            print('Epoch: [{}][{}/{}], Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, num_iter_per_epoch, ave_acc, ave_total_loss ))
            
            fractional_epoch = epoch - 1 + i/num_iter_per_epoch
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())


def checkpoint(nets, history, epoch, train_only_wall=False): 
    """
        Function for saving encoder and decoder weights into a file
    """
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    
    # Depening whether model classifies 2 classes (wall/no wall) or 150 classes, filename changes
    if train_only_wall:
        torch.save(history,      'ckpt/wall_history_epoch_{}.pth'.format(epoch))
        torch.save(dict_encoder, 'ckpt/wall_encoder_epoch_{}.pth'.format(epoch))
        torch.save(dict_decoder, 'ckpt/wall_decoder_epoch_{}.pth'.format(epoch))
    else:
        torch.save(history,      'ckpt/history_epoch_{}.pth'.format(epoch))
        torch.save(dict_encoder, 'ckpt/encoder_epoch_{}.pth'.format(epoch))
        torch.save(dict_decoder, 'ckpt/decoder_epoch_{}.pth'.format(epoch))


def group_weight(module):
    """
        Function for grouping weights and biases of a network into individual groups for training
    """
    group_decay = []
    group_no_decay = []

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.modules.conv._ConvNd)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    return [
        dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=0),
    ]


def create_optimizers(nets):
    """
        Creating individual optimizers for encoder and decoder
    """
    (net_encoder, net_decoder, crit) = nets
    
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder), lr=0.02, momentum=0.9, weight_decay=1e-4 )
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder), lr=0.02, momentum=0.9, weight_decay=1e-4 )
        
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, start_lr=0.02, total_num_iter=1e5 ):
    """
        Adjusting learning rate in each iteration
    """
    scale_running_lr = ((1 - cur_iter/total_num_iter)** 0.9)
    lr_encoder = start_lr * scale_running_lr
    lr_decoder = start_lr * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = lr_decoder


def main_train(last_epoch_trained=0):
    """
        Main function for training original encoder/decoder architecture for semantic segmentation of 150 classes,
        with an option to start from pretrained model, trained in last_epoch_trained
        TODO: Change the name of this function, to be more clear what each of these train functions does
    """
    # Network Builders
    if last_epoch_trained == 0: # when the training starts from nothing
        net_encoder = build_encoder(pretrained=False)
        net_decoder = build_decoder(pretrained=False, use_softmax=False)
        
    else: # When training starts from the previous version of the model
        net_encoder = build_encoder(pretrained=True, epoch=last_epoch_trained)
        net_decoder = build_decoder(pretrained=True, epoch=last_epoch_trained, use_softmax=False)
        
    # Creating criterion. In the dataset there are labels -1 which stand for "don't care", so should be ommited during training.
    crit = nn.NLLLoss(ignore_index=-1) 
    
    # Creating Segmentation Module
    segmentation_module = SegmentationModule(net_encoder, net_decoder).to('cuda')
    
    # Dataset and Loader
    dataset_train = TrainDataset("data/", "data/training.odgt", batch_per_gpu=2)
    
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=16,
        drop_last=True,
        pin_memory=True)

    # create loader iterator
    iterator_train = iter(loader_train)

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets)

    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}
    
    print('Starting training')
    
    # Main loop of certain number of epochs
    for epoch in range(last_epoch_trained, 20):
        train_one_epoch(segmentation_module, iterator_train, optimizers, history, epoch+1, crit)
        checkpoint(nets, history, epoch+1)

    print('Training Done!')


def main_train_wall(train_all=False, train_decoder_all=False, start_lr=0.02, last_epoch_trained=0):
    """
        Main function for training encoder/decoder architecture for semantic segmentation of only wall
        TODO: Change the name of this function, to be more clear what each of these train functions does
    """
    
    if last_epoch_trained > 0: # When training starts from the previous version of the model
        net_encoder = build_encoder(pretrained=True, epoch=last_epoch_trained, train_only_wall=True)
        net_decoder = build_decoder(pretrained=True, epoch=last_epoch_trained,
                                     use_softmax=False, train_only_wall=True)
    else:
        if train_all:
            # Creates an encoder and a decoder where all parameters are trained
            net_encoder = build_encoder(pretrained=False)
            net_decoder = build_decoder(pretrained=False, use_softmax=False)
        else:
            # Loads a pretrained model of an encoder
            net_encoder = build_encoder(pretrained=True, epoch=20)
            for param in net_encoder.parameters():
                param.requires_grad = False

            if train_decoder_all:
                # Creates a decoder where all parameters are trained
                net_decoder = build_decoder(pretrained=False, use_softmax=False)
            else:
                # Loads a pretrained decoder
                net_decoder = build_decoder(pretrained=True, epoch=20, use_softmax=False)
                for param in net_decoder.parameters():
                    param.requires_grad = False
                    
        # Last layer of the decoder is trained
        net_decoder.conv_last[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
    
    # Creating Segmentation Module
    segmentation_module = SegmentationModule(net_encoder, net_decoder).to('cuda')
    
    # Dataset and Loader
    dataset_train = TrainDataset("data/", "data/training.odgt", batch_per_gpu=2, train_only_wall=True)
    
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=16,
        drop_last=True,
        pin_memory=True)
    
    # Creating criterion. In the dataset there are labels -1 which stand for "don't care/background", so should be ommited during training
    crit = nn.NLLLoss(ignore_index=-1, weight=torch.tensor([1, 1], dtype=torch.float).to('cuda'))
    
    # create loader iterator
    iterator_train = iter(loader_train)
    
    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}
        
    # Main loop of certain number of epochs
    print('Starting wall training:')
    for epoch in range(last_epoch_trained, 20):
        train_one_epoch(segmentation_module, iterator_train, optimizers, history, epoch+1, crit, start_lr=start_lr)
        checkpoint(nets, history, epoch+1, train_only_wall=True)

    print('Wall Learning training Done!')