#**Behavioral Cloning** 

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points

<b>Required files:</b>
<table>
	<tr>
        <th>Criteria</th>
        <th>Meets Specifications</th>
    <tr>
    <tr>
        <td>Are all required files submitted?</td>
        <td>The submission includes a model.py file, drive.py, model.h5 a writeup report and run16.mp4</td>
    </tr>
</table> 

<b>Quality of Code:</b>
<table>
	<tr>
        <th>Criteria</th>
        <th>Meets Specifications</th>
    <tr>
    <tr>
        <td>Is the code functional?</td>
        <td>The model provided can be used to successfully operate the simulation.</td>
    </tr>
	<tr>
		<td>Is the code usable and readable?</td>
		<td>The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.</td>
	</tr>
</table> 

<b>Model Architecture and Training Strategy:</b>
<table>
	<tr>
        <th>Criteria</th>
        <th>Meets Specifications</th>
    <tr>
    <tr>
        <td>Has an appropriate model architecture been employed for the task?</td>
        <td>The architecture implemented was NVIDEA because it is the most suitable for this kind of task, with more accuracy than the others networks</td>
    </tr>
	<tr>
		<td>Has an attempt been made to reduce overfitting of the model?</td>
		<td>Avoiding the overfitting was implemented 5 dropouts, after the first layer a dropout of 59%, after the second layer a dropout of 49%, after the third layer a dropout of 39%, after the fourth layer a dropout of 29% and then after the fifth layer a dropout of 19%, and as well as was implemented ELU after every dropout</td>
	</tr>
	<tr>
		<td>Have the model parameters been tuned appropriately?</td>
		<td>Learning rate parameters were choosen after the observation of the behavior of every run and as well was use ADAM in the deep learning network</td>
	</tr>
	<tr>
		<td>Is the training data chosen appropriately?</td>
		<td>The training data choose was the data from udacity, because the author is not accurate enough when is driving the simulator</td>
	</tr>			
</table> 

<b>Architecture and Training Documentation</b>
<table>
	<tr>
        <th>Criteria</th>
        <th>Meets Specifications</th>
    <tr>
    <tr>
        <td>Is the solution design documented?</td>
        <td>TThe README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.</td>
    </tr>
	<tr>
		<td>Is the model architecture documented?</td>
		<td>The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.</td>
	</tr>
	<tr>
		<td>Is the creation of the training dataset and training process documented?</td>
		<td>The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included.</td>
	</tr>		
</table> 

<b>Simulation:</b>
<table>
	<tr>
        <th>Criteria</th>
        <th>Meets Specifications</th>
    <tr>
    <tr>
        <td>Is the car able to navigate correctly on test data?</td>
        <td>No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).</td>
    </tr>
</table> 

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is the implementation of the neural network of NVIDEA, this network consist in

________________________________¦____________________¦__________¦______________________________________
Layer (type)                     Output Shape          Param #     Connected to
================================¦====================¦==========¦======================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 70, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 33, 158, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 33, 158, 24)   0           dropout_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       elu_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 15, 77, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 15, 77, 36)    0           dropout_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       elu_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 6, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 6, 37, 48)     0           dropout_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       elu_3[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 4, 35, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 4, 35, 64)     0           dropout_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       elu_4[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 2, 33, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 2, 33, 64)     0           dropout_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4224)          0           elu_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           422500      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
====================================================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
____________________________________________________________________________________________________

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
