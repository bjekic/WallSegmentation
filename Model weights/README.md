Inside this folder there are weights for the trained models: <br/>
 - [resnet50-imagenet](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/resnet50-imagenet.pth) - A pretrained model of the ResNet-50 architecture. Pretrained on ImageNet database. <br/>
 
 - Weights for the segmentation module trained for semantic segmentation of 150 different classes found in the ADE20K database: <br/>
   - [encoder_epoch_20](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/encoder_epoch_20.pth) - Weights for the encoder. <br/>
   - [decoder_epoch_20](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/decoder_epoch_20.pth) - Weights for the decoder. <br/>
   
 - Weights for the segmentation module after transfer learning for semantic segmentation of walls. Transfer learning is done only for the last layer of the decoder architecture: <br/>
   - [Output_only_encoder](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/Output_only_encoder.pth) -  Weights for the encoder. <br/>
   - [Output_only_decoder](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/Output_only_decoder.pth) -  Weights for the decoder. <br/>
   
 - Weights for the segmentation module after transfer learning for semantic segmentation of walls. Transfer learning is done for the entire decoder architecture: <br/>
   - [transfer_encoder](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/transfer_encoder.pth) - Weights of the encoder. <br/>
   - [transfer_decoder](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/transfer_decoder.pth) - Weights of the decoder. <br/>
   
 - Weights for the segmentation module trained for semantic segmentation of only walls (The training is done on a subset of the ADE20K database): <br/>
   - [wall_encoder_epoch_20](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/wall_encoder_epoch_20.pth) - Weights of the encoder. <br/>
   - [wall_decoder_epoch_20](https://github.com/bjekic/WallSegmentation/blob/main/Model%20weights/wall_decoder_epoch_20.pth) - Weights of the decoder. <br/>
