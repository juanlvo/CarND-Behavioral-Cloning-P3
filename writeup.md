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

[image1]: nvidia.png "Model Visualization NVIDIA"
[image2]: center.png "Center camera in the car"
[image3]: left.png "Left camera in the car"
[image4]: right.png "Right camera in the car"
[image5]: figure_1_16.png "Training graph"


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

<table>
	<tr>
		<th>Layer (type)</th>
		<th>Output Shape</th>
		<th>Param #</th>
		<th>Connected to</th>
		<th>Description</th>
	</tr>
	<tr>
		<td>lambda_1 (Lambda)</td>
		<td>(None, 160, 320, 3)</td>
		<td>0</td>
		<td>lambda_input_1[0][0]</td>
		<td></td>
	</tr>
	<tr>
		<td>cropping2d_1 (Cropping2D)</td>
		<td>(None, 70, 320, 3)</td>
		<td>0</td>
		<td>lambda_1[0][0]</td>
		<td>Cropping the image for generalize more the model</td>
	</tr>
	<tr>
		<td>convolution2d_1 (Convolution2D)</td>
		<td>(None, 33, 158, 24)</td>
		<td>1824</td>
		<td>convolution2d_1[0][0]</td>
		<td>First Convolutional layer (24x5x5) describe it by NVIDEA arquitecture</td>
	</tr>
	<tr>
		<td>dropout_1 (Dropout)</td>
		<td>(None, 33, 158, 24)</td>
		<td>0</td>
		<td>convolution2d_1[0][0]</td>
		<td>Dropout of 59% avoiding overfiting</td>
	</tr>
	<tr>
		<td>elu_1 (ELU)</td>
		<td>(None, 33, 158, 24)</td>
		<td>0</td>
		<td>dropout_1[0][0]</td>
		<td>ELU avoiding overfiting</td>
	</tr>
	<tr>
		<td>convolution2d_2 (Convolution2D)</td>
		<td>(None, 15, 77, 36)</td>
		<td>21636</td>
		<td>elu_1[0][0]</td>
		<td>Second Convolutional layer (36x5x5) describe it by NVIDEA arquitecture</td>		
	</tr>
	<tr>
		<td>dropout_2 (Dropout)</td>
		<td>(None, 15, 77, 36)</td>
		<td>0</td>
		<td>convolution2d_2[0][0]</td>
		<td>Dropout of 49% avoiding overfiting</td>
	</tr>
	<tr>
		<td>elu_2 (ELU)</td>
		<td>(None, 15, 77, 36)</td>
		<td>0</td>
		<td>dropout_2[0][0]</td>
		<td>ELU avoiding overfiting</td>
	</tr>
	<tr>
		<td>convolution2d_3 (Convolution2D)</td>
		<td>(None, 6, 37, 48)</td>
		<td>43248</td>
		<td>elu_2[0][0]</td>
		<td>Third Convolutional layer (48x5x5) describe it by NVIDEA arquitecture</td>		
	</tr>
	<tr>
		<td>dropout_3 (Dropout)</td>
		<td>(None, 6, 37, 48)</td>
		<td>0</td>
		<td>convolution2d_3[0][0]</td>
		<td>Dropout of 39% avoiding overfiting</td>
	</tr>
	<tr>
		<td>elu_3 (ELU)</td>
		<td>(None, 6, 37, 48)</td>
		<td>0</td>
		<td>dropout_3[0][0]</td>
		<td>ELU avoiding overfiting</td>
	</tr>
	<tr>
		<td>convolution2d_4 (Convolution2D)</td>
		<td>(None, 4, 35, 64)</td>
		<td>27712</td>
		<td>elu_3[0][0]</td>
		<td>Fourth Convolutional layer (64x3x3) describe it by NVIDEA arquitecture</td>		
	</tr>
	<tr>
		<td>dropout_4 (Dropout)</td>
		<td>(None, 4, 35, 64)</td>
		<td>0</td>
		<td>convolution2d_4[0][0]</td>
		<td>Dropout of 29% avoiding overfiting</td>
	</tr>
	<tr>
		<td>elu_4 (ELU)</td>
		<td>(None, 4, 35, 64)</td>
		<td>0</td>
		<td>dropout_4[0][0]</td>
		<td>ELU avoiding overfiting</td>
	</tr>
	<tr>
		<td>convolution2d_5 (Convolution2D)</td>
		<td>(None, 2, 33, 64)</td>
		<td>36928</td>
		<td>elu_4[0][0]</td>
		<td>Fifth Convolutional layer (64x3x3) describe it by NVIDEA arquitecture</td>		
	</tr>
	<tr>
		<td>dropout_5 (Dropout)</td>
		<td>(None, 2, 33, 64)</td>
		<td>0</td>
		<td>convolution2d_5[0][0]</td>
		<td>Dropout of 19% avoiding overfiting</td>
	</tr>
	<tr>
		<td>elu_5 (ELU)</td>
		<td>(None, 2, 33, 64)</td>
		<td>0</td>
		<td>dropout_5[0][0]</td>
		<td>ELU avoiding overfiting</td>
	</tr>
	<tr>
		<td>flatten_1 (Flatten)</td>
		<td>(None, 4224)</td>
		<td>0</td>
		<td>elu_5[0][0]</td>
		<td>Flatten layer describe it by NVIDEA arquitecture</td>
	</tr>
	<tr>
		<td>dense_1 (Dense)</td>
		<td>(None, 100)</td>
		<td>422500</td>
		<td>flatten_1[0][0]</td>
		<td>Dense layer of 100 describe it by NVIDEA arquitecture</td>
	</tr>
	<tr>
		<td>dense_2 (Dense)</td>
		<td>(None, 50)</td>
		<td>5050</td>
		<td>dense_1[0][0]</td>
		<td>Dense layer of 50 describe it by NVIDEA arquitecture</td>
	</tr>
	<tr>
		<td>dense_3 (Dense)</td>
		<td>(None, 10)</td>
		<td>510</td>
		<td>dense_2[0][0]</td>
		<td>Dense layer of 10 describe it by NVIDEA arquitecture</td>
	</tr>
	<tr>
		<td>dense_4 (Dense)</td>
		<td>(None, 1)</td>
		<td>11</td>
		<td>dense_3[0][0]</td>
		<td>Dense layer of 1 describe it by NVIDEA arquitecture</td>
	</tr>
</table>


Total params: 559,419<br>
Trainable params: 559,419<br>
Non-trainable params: 0<br>


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 107, 110, 113, 116, 119) and as well ELU (lines 108, 111, 114, 117, 120)

The model was trained and validated on dataset provided by Udacity. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 127).

####4. Appropriate training data

The training dataset was provided by Udacity.com because is more suitable than the data generated by the author of this project.


###Model Architecture and Training Strategy

####1. Solution Design Approach


The overall strategy for deriving a model architecture was to try with the different types of models propose in the course (LeNet, Lambda and NVIDIA)

With LeNet and Lambda was not really accurate when was driving the autonmous car, but NVIDIA was more accurate at the starting of the driving, this was a good sign for starting, that's why was choose this type of architecture.

But after of this good start was need it to improve the accurate so first was added a dropout after every convolutional network and changing the values of every dropout until realize the best version was 59%, 49%, 39%, 29% and 19%, The decision of these numbers was complety random after several sessions of testing.

Then after to improve the model was added ELUs after every dropout and finally we got a acceptable model for driving the track 1


####2. Final Model Architecture

The final model architecture (model.py lines 103-130) is the implementation of NVIDIA model which is consisted by:

1. Convolutional Layer 24x5x5
2. Convolutional Layer 36x5x5
3. Convolutional Layer 48x5x5
4. Convolutional Layer 64x3x3
5. Convolutional Layer 64x3x3
6. Flatten Layer
7. Dense Layer 100
8. Dense Layer 50
9. Dense Layer 10
10. Dense Layer 1

Here is a visualization of the architecture

![alt text][image1]

####3. Dataset provided by udacity

Here are some examples of the dataset use by the solution

![alt text][image2]
![alt text][image3]
![alt text][image4]

As you can see was use it the 3 cameras with a creection of +/- 0.2 in the angle 

To augment the data sat, was also flipped images and angles to have more accuracy from the model.

This data set was randomly shuffled and putted Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 as evidenced here is a graphic of the training

![alt text][image5]
