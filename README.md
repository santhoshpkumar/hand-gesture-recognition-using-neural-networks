# Hand gesture recognition using neural networks

## Problem Statement

Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

- Thumbs up:  Increase the volume
- Thumbs down: Decrease the volume
- Left swipe: 'Jump' backwards 10 seconds
- Right swipe: 'Jump' forward 10 seconds  
- Stop: Pause the movie

Each video is a sequence of 30 frames (or images)

## Understanding the Dataset

The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people performing one of the five gestures in front of a webcam - similar to what the smart TV will use.

The data is in a [zip](https://www.dropbox.com/s/72jzl3fqvk1rk2w/gesture_data.zip?dl=0) file. The zip file contains a 'train' and a 'val' folder with two CSV files for the two folders.

![dataset](images/dataset1.png)

These folders are in turn divided into subfolders where each subfolder represents a video of a particular gesture.

![dataset](images/dataset2.png)

Each subfolder, i.e. a video, contains 30 frames (or images). 

- Thumbs Up
  
![dataset](images/gesture_thumbs_up.png)

- Right Swipe

![dataset](images/gesture_right_swipe.png)

Note that all images in a particular video subfolder have the same dimensions but different videos may have different dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used to record the videos).

## Two Architectures: 3D Convs and CNN-RNN Stack

After understanding and acquiring the dataset, the next step is to try out different architectures to solve this problem. 

For analysing videos using neural networks, two types of architectures are used commonly. 

One is the standard **CNN + RNN** architecture in which you pass the images of a video through a CNN which extracts a feature vector for each image, and then pass the sequence of these feature vectors through an RNN. 

*Note:*
 - You can use transfer learning in the 2D CNN layer rather than training your own CNN 
 - GRU (Gated Recurrent Unit) or LSTM (Long Short Term Memory) can be used for the RNN

The other popular architecture used to process videos is a natural extension of CNNs - a **3D convolutional network**. In this project, we will try both these architectures.


# Implementation 

## 3D Convolutional Network, or Conv3D

Now, lets implement a 3D convolutional Neural network on this dataset. To use 2D convolutions, we first convert every image into a 3D shape : width, height, channels. Channels represents the slices of Red, Green, and Blue layers. So it is set as 3. In the similar manner, we will convert the input dataset into 4D shape in order to use 3D convolution for : length, breadth, height, channel (r/g/b).

*Note:* even though the input images are rgb (3 channel), we will perform image processing on each frame and the end individual frame will be grayscale (1 channel)

Lets create the model architecture. The architecture is described below:

Input and Output layers:

- One Input layer with dimentions 160, 160, 30, 1
- Output layer with dimentions 5

Convolutions :

- Apply 4 Convolutional layer with increasing order of filter size (standard size : 8, 16, 32, 64) and fixed kernel size = (3, 3, 3)
- Apply 2 Max Pooling layers, one after 2nd convolutional layer and one after fourth convolutional layer.

MLP (Multi Layer Perceptron) architecture:

- Batch normalization on convolutiona architecture
- Dense layers with 2 layers followed by dropout to avoid overfitting

```python
## input layer
input_layer = Input((150, 150, 30, 3))

## convolutional layers
conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(conv_layer1)

## add max pooling to obtain the most imformatic features
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(pooling_layer1)
conv_layer4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv_layer3)
pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)

## perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer2 = BatchNormalization()(pooling_layer2)
flatten_layer = Flatten()(pooling_layer2)

## create an MLP architecture with dense layers :  500 -> 50 -> 5
## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(units=500, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=50, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=5, activation='softmax')(dense_layer2)

## define the model with input layer and output layer
model = Model(inputs=input_layer, outputs=output_layer)
```

## Data Preprocessing

We need to covert the color images to grayscale images have pixel values that range from 0 to 255. Also, these images have varied dimension 360*360 and 120*160. As a result, you'll need to preprocess the data before you feed it into the model.

As a first step, convert each image of the train and test set into a matrix of size 120*120, followed by some image thresholding and finally to a grayscale image.