# Wall Segmentation

Implementation of a wall segmentation algorithm in PyTorch. Implementation is based on a [paper](https://arxiv.org/abs/1612.01105).<br/> 

The used database is [MIT ADE20K Scene parsing dataset](http://sceneparsing.csail.mit.edu/), where 150 different categories are labeled.
An example of an image from the database: ![Example]( url u repository )<br/> <br/> 

Because for solving the problem of wall segmentation, we do not need all the images inside the ADE20K database (we need only indoor images), a subset of the database is used in
training the Model.<br/> <br/>

##Segmentation architecture:<br/> 
 - Encoder - obtains a feature map of the original image that has smaller height and width anda larger number of channels. The architecture of the encoder, used in this
project is the Dilated ResNet 50, where the last two blocks of the ResNet50 architecture use dilated convolution with a smaller stride. <br/> 
 - Decoder - Based on the feature map, classifies each pixel of the feature map into one of the classes. The architecture of the decoder, used in this project is the 
PPM architecture.<br/> <br/> 

## Structure of the project:<br/>
 - folder [Models](https://github.com/bjekic/WallSegmentation/tree/main/Models) consists of 3 seperate .py files: <br/> 
   - [resnet.py](https://github.com/bjekic/WallSegmentation/blob/main/Models/resnet.py) - where the ResNet architecture is defined. <br/>
   - [models.py](https://github.com/bjekic/WallSegmentation/blob/main/Models/models.py) - where the whole PPM architecture for the decoder is defined, the ResNet dilated architecture for the
encoder, as well as the class for the Segmentation Module. Beside the classes for the different architectures, there are
2 helper functions for instantiating the encoder network as well as the decoder network.<br/>
   - [dataset.py](https://github.com/bjekic/WallSegmentation/blob/main/Models/dataset.py) - where the derived classes of the abstract class [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html), are defined. 
Two defined derived classes are a class for loading the training data, and a class for loading the validation data. The implementation of the TrainDataset and ValDataset are
are taken from [CSAILVision/semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) and small changes were made. The changes made to the 
original implementations of TrainDataset and ValDataset are for loading only images of interest (images that contain wall segments). Also, inside this folder, 
an additional function, for differentiating between the images of interest and other images, is implemented. <br/>
 - folder [Model weights]( url ) - where weights of the trained models are held.<br/>
 - folder [data](https://github.com/bjekic/WallSegmentation/tree/main/data) - where the database is held, as well as files used for loading the dataset.  (The Database is not present in the directory due to size) <br/>
 - folder [cktp]( url ) - where checkpoints during training of the models are saved. (Because the models are trained, the directory is now empty)<br/>
 - [train.py](https://github.com/bjekic/WallSegmentation/blob/main/train.py) - file where helper functions for training the segmentation module are implemented.<br/>
 - [eval.py](https://github.com/bjekic/WallSegmentation/blob/main/eval.py) - file where helper functions for evaluating the segmentation module are implemented. <br/>
 - [Model_training.ipynb](https://github.com/bjekic/WallSegmentation/blob/main/Model_training.ipynb) - the code for training the segmentation module on 150 different classes of the ADE20K dataset.<br/>
 - [Only_wall_training.ipynb](https://github.com/bjekic/WallSegmentation/blob/main/Only_wall_training.ipynb) - the code for training the segmentation module for segmenting only wall in the images of interest.
There are different models for training the segmentation module, described in detail in the code. <br/>
 - [Testing.ipynb](https://github.com/bjekic/WallSegmentation/blob/main/Testing.ipynb) - the code for testing the segmentation module on the validation dataset, as well as to show the results of wall segmentation on any RGB image.<br/><br/>



##Training of the segmentation module<br/>

The main contibution of this project is simplifying the project already available at the [link](https://github.com/CSAILVision/semantic-segmentation-pytorch) and training 
the segmentation module only for wall segmentation. Three different approaches to training the model were considered. In all three approaches, the weights of the encoder are
initialized using the pretrained model of ResNet50 trained on the ImageNet database. First considered approach was to use transfer learning,
to train the model on all 150 different categories, and then to change the output layer of the decoder and train only the output layer on images of interest. The second approach
differs from the first approach only in the last part, where not only the last layer is trained on images of interest, instead, the entire decoder is trained on images of interes. The third
approach does not include transfer learning as described in previous two cases. Instead, the segmentation module is trained from the start on the images of interest. 

## Results<br/>

Mean values of pixel accuracy and IoU on validation subset are given in the table.

|                  |  First approach  | Second approach  |  Third approach  |
|:----------------:|:----------------:|:----------------:|:----------------:|
|Pixel accuracy [%]|      84.82       |      86.24       |      90.39       |
|IoU [%]           |      56.87       |      59.08       |      68.79       |


![Result obtained using first approach]( url u repository )<br/> <br/> 
![Result obtained using second approach]( url u repository )<br/> <br/>
![Result obtained using third approach]( url u repository )<br/> <br/>
[From left to right: Test image, Segmentation mask, Predicted result]
