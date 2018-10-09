# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]: ./examples/nvidia.png "NVIDIA"
[center]: ./examples/center.jpg "Center"
[left]: ./examples/left.jpg "Left"
[right]: ./examples/right.jpg "Right"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with filter sizes between 3x3 and 5x5, and depths between 24 and 64 (L73-87) based on the NVIDIA architecture.

![NVIDIA][nvidia]

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).
The original input image shape is `(160, 320, 3)`, which is normalized using a `Lambda` layer (L74), and then cropped using a `Cropping2D` layer to trim out the sky and the hood of the car, resulting in a `(75, 320, 3)` input shape. The NVIDIA architecture describes using YUV, so in the batch generator the images are converted to YUV colorspace (L48), though in testing this didn't seem to make much of a difference from using RGB.

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, the model contains a dropout layer after the CNN layers (L81). In addition, the data is shuffled and augmented each epoch (L22), see the description of the augmentation pipeline in the "Creation of the Training Set & Training Process" section.

#### 3. Model parameter tuning

The model used the `Adam` optimizer, which automatically tunes the learning rate (L92). To select the best model over the training epochs, the model uses `ModelCheckpoint` (L93), which saves the models with the lowest validation set loss.

#### 4. Appropriate training data

I used the training data that was provided by Udacity without any additional training of my own, though I used data augmentation on the training set, see the next section for more details. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to try and get Keras working, so I had a minimal model architecture with only a `Flatten` and `Dense(1)` layer. Predictably, this drove very poorly, but now that the code was working, I implemented the CNN architecture described in the NVIDIA article [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/), which was successfully used by their team to train a self-driving car. I made a few changes, one of which was using ELU activation to introduce nonlinearity, as it seems to be the new [hotness](https://sefiks.com/2018/01/02/elu-as-a-neural-networks-activation-function/) and has shown to decrease training time. In addition, I added a dropout layer after the convolution layers in order to reduce overfitting.

#### 2. Final Model Architecture

Here is my implementation of the NVIDIA architecture with a few changes:

| Layer                 |     Description	        					            | 
|:---------------------:|:---------------------------------------------:| 
| Input         		    | 75x320 YUV image   			  		                | 
| Convolution 5x5     	| 24 filters, 2x2 stride                        |
| ELU					          |												                        |
| Convolution 5x5     	| 36 filters, 2x2 stride                     	  |
| ELU					          |												                        |
| Convolution 5x5     	| 48 filters, 2x2 stride                     	  |
| ELU					          |												                        |
| Convolution 3x3     	| 64 filters, 1x1 stride                     	  |
| ELU					          |												                        |
| Convolution 3x3     	| 64 filters, 1x1 stride                     	  |
| ELU					          |												                        |
| Dropout				        | Keep percentage = 0.5						            	|
| Flatten			          |           				 					                  |
| Fully Connected		    | Outputs 100                 									|
| Fully Connected		    | Outputs 50                   									|
| Fully Connected		    | Outputs 10                   									|
| Fully Connected		    | Outputs 1                   									|

#### 3. Creation of the Training Set & Training Process

I reserved 20% of the image and steering angle data for the validation set (L13). After training for 3 epochs with a batch size of 32, the test and validation error were more or less the same, so I saved the model and tested it with the simulator. The car did fairly well around the first two turns, but after the bridge, it drove off the curb and could not recover.

Since the training data was biased towards center of the lane driving, I needed more training examples for driving at the edges. The training data includes left and right camera images, but to add a little more bias, I selected the camera perspective based on the steering angle. For example, if the steering angle was above 0.15 (which means the car is turning left-to-right), there was a 50/50 chance of choosing either the left camera closest to the lane edge or the center camera. In addition, depending on the camera, an angle correction factor of 0.25 was added or subtracted. Finally, the image was randomly flipped horizontally to further augment the data.

Sample center camera image:
![Center camera][center]

Sample left camera image at the edge:
![Left camera][left]

Sample right camera image at the edge:
![Right camera][right]

At the end of the training process, the vehicle was able to successfully drive around the track without driving off the road.
