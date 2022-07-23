# Model weights

Inside this folder models for the trained models are stored. Due to the size of the models, the models can be found
on [link](https://drive.google.com/drive/folders/1xh-MBuALwvNNFnLe-eofZU_wn8y3ZxJg?usp=sharing):

 - resnet50-imagenet.pth - A pretrained model of the ResNet-50 architecture. Pretrained on ImageNet database,

 - Weights for the segmentation module trained for semantic segmentation of 150 different classes found in the 
ADE20K database (directory: Pretrained model on 150 classes):
   - encoder_epoch_20.pth - Weights for the encoder after training the model for 20 epochs,
   - decoder_epoch_20.pth - Weights for the decoder after training the model for 20 epochs.

 - Weights for the segmentation module after transfer learning for semantic segmentation of walls. Transfer learning is
done only for the last layer of the decoder architecture (directory: Transfer learning - last layer):
   - Output_only_encoder.pth -  Weights for the encoder,
   - Output_only_decoder.pth -  Weights for the decoder.
   
 - Weights for the segmentation module after transfer learning for semantic segmentation of walls. Transfer learning is
done for the entire decoder architecture (directory: Transfer learning - entire decoder):
   - transfer_encoder.pth - Weights of the encoder,
   - transfer_decoder.pth - Weights of the decoder.
   
 - Weights for the segmentation module trained for semantic segmentation of only walls (The training is done on a subset 
of the ADE20K database) (directory: Without transfer learning):
   - best_encoder_epoch_19.pth - Weights of the encoder,
   - best_decoder_epoch_19.pth - Weights of the decoder.
