import json
import os
import shutil
import torch
import torch.nn as nn
from Models.models import SegmentationModule, build_encoder, build_decoder
from Models.dataset import TrainDataset
from Utils.constants import NUM_EPOCHS, OPTIMIZER_PARAMETERS, DEVICE, NUM_WORKERS, ODGT_TRAINING, BATCH_PER_GPU
from torch.utils.tensorboard import SummaryWriter
from src.train import create_optimizers, train_one_epoch, checkpoint
from src.eval import main_evaluate
import pickle


def stupid_collate_fn(x):
    return x


def main_train(ckpt_dir_path,
               data_root_path,
               continue_training=False,
               path_encoder_weights="",
               path_decoder_weights=""):
    """
        Main function for training original encoder/decoder architecture for semantic segmentation of 150 classes,
        with an option to start from pretrained model, trained in last_epoch_trained
        TODO: Change the name of this function, to be more clear what each of these train functions does
    """
    # Encoder/Decoder weights
    if continue_training:
        last_epoch = [int(x.split('.')[0].split('_')[-1]) for x in os.listdir(ckpt_dir_path) if x.startswith('encoder_epoch_')][0]
        path_encoder_weights = os.path.join(ckpt_dir_path, f'encoder_epoch_{last_epoch}.pth')
        path_decoder_weights = os.path.join(ckpt_dir_path, f'decoder_epoch_{last_epoch}.pth')
        print(f"The training will continue from {last_epoch + 1} epoch...")
    else:
        last_epoch = 0
        if os.path.exists(ckpt_dir_path):
            shutil.rmtree(ckpt_dir_path)
        os.mkdir(ckpt_dir_path)

    net_encoder = build_encoder(path_encoder_weights)
    net_decoder = build_decoder(path_decoder_weights, use_softmax=False)

    # Creating criterion. In the dataset there are labels -1 which stand for "don't care", so should be ommited during training.
    crit = nn.NLLLoss(ignore_index=-1)

    # Creating Segmentation Module
    segmentation_module = SegmentationModule(net_encoder, net_decoder).to(DEVICE)

    # Dataset and Loader
    dataset_train = TrainDataset(data_root_path, ODGT_TRAINING, batch_per_gpu=BATCH_PER_GPU)

    loader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=1, # TODO: write why it is one (because the batch is created in TrainDataset)
                                               shuffle=False,
                                               collate_fn=stupid_collate_fn,
                                               num_workers=NUM_WORKERS, # TODO change to parameter from config.json/contants.py
                                               drop_last=True,
                                               pin_memory=True)

    # create loader iterator
    iterator_train = iter(loader_train)

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, OPTIMIZER_PARAMETERS)

    # Tensorboard initialization
    writer = SummaryWriter(os.path.join(ckpt_dir_path, 'tensorboard'))

    print('Starting training')
    # Main loop of certain number of epochs
    path_train_metadata = os.path.join(ckpt_dir_path, 'training_metadata.pkl')
    if os.path.exists(path_train_metadata):
        with open(path_train_metadata, 'rb') as f:
            train_metadata = pickle.load(f)
    else:
        train_metadata = {'best_acc': 0, 'best_IOU': 0}

    for epoch in range(last_epoch, NUM_EPOCHS):
        print(f'Training epoch {epoch + 1}/{NUM_EPOCHS}...')
        train_one_epoch(segmentation_module, iterator_train, optimizers, epoch + 1, crit, writer)
        acc, IOU = main_evaluate(segmentation_module, data_root_path, writer, epoch+1)
        if acc > train_metadata['best_acc']:
            train_metadata['best_acc'] = acc
            train_metadata['best_IOU'] = IOU
            save_best = True
            with open(path_train_metadata, 'wb') as f:
                pickle.dump(train_metadata, f)
            print(f'Epoch {epoch + 1} is the new best epoch!')
        else:
            save_best = False
        checkpoint(nets, epoch + 1, ckpt_dir_path, save_best)

    writer.close()
    print('Training Done!')


if __name__ == '__main__':

    with open('configs/config_desktop_everseen.json', 'r') as f:
        config = json.load(f)

    main_train(ckpt_dir_path=config["CHECKPOINT_DIR_PATH"],
               data_root_path=config["ROOT_DATASET"],
               continue_training=config["CONTINUE_TRAINING"],
               path_encoder_weights=config["MODEL_ENCODER_WEIGHTS_PATH"],
               path_decoder_weights=config["MODEL_DECODER_WEIGHTS_PATH"])