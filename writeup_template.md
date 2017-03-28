#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/manujshinkar/Traffic-Sign-Classifier)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data of number of images for each traffic sign in the training dataset.

![alt tag](https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/train_distribution.png)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I normalize the image data in order to make the problem well conditioned so that the optimizer will run faster.

Here is an example of a traffic sign image before and after normalizing.

<img src="https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/before_normalization.png" width="200" height="200" />
<img src="https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/before_normalization.png" width="200" height="200" />

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for getting the data into training, testing and validation sets is contained in the first code cell of the IPython notebook.  

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU		            |        								    	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten				| outputs vector of length 400					|
| Fully Connected		| mean = 0 sigma = 0.1 outputs 120				|
| RELU		            |        								    	|
| Dropout		        | keep Prob = 0.5        					    |
| Fully Connected		| mean = 0 sigma = 0.1 outputs 84				|
| RELU		            |        								    	|
| Dropout		        | keep Prob = 0.5        					    |
| Fully Connected		| mean = 0 sigma = 0.1 outputs 43				|
| Softmax			    | Cross Entropy									|
| Reduce Mean			| Loss Operation								|
| Adam Optimizer	    | Minimize loss with leaning rate 0.001			|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an adam optimizer with batch size of 128 and learning rate of 0.001. The model was trained for 20 epochs. Variables were initialized using truncated normal distribution with mean 0 and standard deviation 0.1. The training took around 10-12 minutes on CPU of my machine.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.991
* validation set accuracy of 0.940
* test set accuracy of 0.932

This project is about traffic sign classification given an image. CNN is best suitable for such problems. I used a Lenet-5 architecture for this project because it is a well known architecture for CNN and I got good results with it. After strating the Lenet architecure it was all about parameter and hyperparameter tuning. Since the training, validation and testing accuracy are almost comparable, my model seems to work fine. Since there are more images for some traffic signs as compared to others, I belive the accuracy can be improoved by adding more data for the traffic signs with less images. The below figure shows the graph for validation accuracy. It bacame almost constant at 0.94.

![alt tag](https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/accuracy.png)  
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/i_1.png" width="200" height="200" />
<img src="https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/i_11.png" width="200" height="200" />
<img src="https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/i_13.png" width="200" height="200" />
<img src="https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/i_35.png" width="200" height="200" />
<img src="https://github.com/manujshinkar/Traffic-Sign-Classifier/blob/master/i_38.png" width="200" height="200" />

The first image might be difficult to classify because it has some shadow on it. 
The second and third images have good exposure.
The fourth image is bright.
The fourth and fifth image have less number of images in the training data set.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my new images is located in the thirteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Speed limit (30km/h) 							| 
| Right-of-way at the next intersection | Right-of-way at the next intersection	|
| Yield					| Yield											|
| Ahead only     		| Yield					 			        	|
| Keep right	    	| Yield    						            	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is less than the accuracy of the test set. I believe that this is because of relatively less samples of the ahead only and keep right signs.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for showing softmax probablities of new images is in the fifteenth cell of the Ipython notebook.

For the first image The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.983         		| Speed limit (30km/h)   						| 
| 0.0137     			| Speed limit (20km/h)	|
| 0.0018				| Speed limit (70km/h)								|
| 0.0001	      			| End of all speed and passing limits		|
| 0.0				    | Speed limit (40km/h)   					    	|

For the second image The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Right-of-way at the next intersection   		| 
| 0.0     				| Beware of ice/snow		|
| 0.0					| Pedestrians										|
| 0.0	      			| Priority road					 				|
| 0.0				    | Double curve    					    		|

For the third image The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield  						| 
| 0.0     				| 	Ahead only	|
| 0.0					| 	Turn left ahead									|
| 0.0	      			| 	Keep left				 				|
| 0.0				    |  Keep right     					    		|

For the fourth image The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.817         			| No passing   						|
| 0.0958     				| No passing for vehicles over 3.5 metric tons	|
| 0.0845					| Yield											|
| 0.0009	      			| Bicycles crossing					 			|
| 0.0009				    | No vehicles     					    		|

For the fifth image The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9995         			| Yield   						|
| 0.0002     				| Road work			|
| 0.0002				| Keep left											|
| 0.0001	      			| Bumpy road					 				|
| 0.0				    | Keep right     					    		|

