
**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize layer of the neural network
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/barplot.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/image_shift.jpg "Image shift"
[image4]: ./examples/image_01.jpg =30x30 "Traffic Sign 1"
[image5]: ./examples/image_02.jpg =30x30 "Traffic Sign 2"
[image6]: ./examples/image_03.jpg =30x30 "Traffic Sign 3"
[image7]: ./examples/image_04.jpg =30x30 "Traffic Sign 4"
[image8]: ./examples/image_05.jpg =30x30 "Traffic Sign 5"
[image9]: ./examples/image_06.png "softmax probalities for image 1"
[image10]: ./examples/image_07.png "softmax probalities for image 2"
[image11]: ./examples/image_08.png "softmax probalities for image 3"
[image12]: ./examples/image_09.png "softmax probalities for image 4"
[image13]: ./examples/image_10.png "softmax probalities for image 5"
[image14]: ./examples/sample_image.png "Sample Image"
[image15]: ./examples/sample_image_gray.png "Sample Image Gray"
[image16]: ./examples/sample_image_gray_shift_up.png "Sample Image Gray shifted up"
[image17]: ./examples/sample_image_gray_shift_down.png "Sample Image Gray shifted down"
[image18]: ./examples/sample_image_gray_shift_left.png "Sample Image Gray shifted left"
[image19]: ./examples/sample_image_gray_shift_right.png "Sample Image Gray shifted right"
[image20]: ./examples/first_layer.png "First Layer"
[image21]: ./examples/second_layer.png "Second Layer"


---

Here is a link to my [project code](https://github.com/Noyin/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

**Steps**

**1. Load Data Set**

I loaded the data set using pickle in the first code cell of the IPython notebook. Here is a link to the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)

**2. Data Set Summary & Exploration**


In the second code cell of the IPython notebook, I used native python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

**3. Exploratory visualization of the dataset**.

The code for this step is contained in the third code cell of the IPython notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images per traffic sign

![alt text][image1]

**4. Design and Test a Model Architecture**


Processing of data is done in the fourth code cell of the Ipython notebook. I decided to convert the images in the dataset to grayscale first to improve training time and also because in
this situation, color is not a deciding factor for correctly categorizing traffics signs. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image14]![alt text][image15]

I further proceeded to normalize the images  make it easier to training a model and to improve training time.

As a last step, I augmented the images by creating a copy of each image and adding shifts(-/+3) in all direction. This increases the training dataset from  34799 to 173995. Adding augmented data helps to train a better model.
Here is an example of a traffic sign image before and after shift is added

![alt text][image15]![alt text][image16]![alt text][image17]![alt text][image18]![alt text][image19]



The code for my model architecture is located in the fifth code cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Gray image   							|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| 												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					| 												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                   |
| Fully connected		| outputs 1X120       						    |
| Fully connected		| outputs 1x84        						    |
| Softmax				| output 43        								|
|						|												|
|						|												|





The code for training the model is located in the sixth code cell of the ipython notebook.

To train the model, I adjusting the values of parameters to get the highest prediction accuracy for my model.The following are the parameters I used to train my final model:

*learning_rate = 0.001
*epochs = 22
*batch_size = 128
*test_valid_size = 256
*dropout =1.00
*mu = 0
*sigma = 0.1

I set the value of epoch to 22 as there was no improvement to the prediction accuracy for additional epochs.


The code for calculating the accuracy of the model is also located in the sixth cell of the Ipython notebook.

To arrive at my final model, I adjusted each parameter one at a time and chose the value that yielded the best results then moved to adjusting the next parameter.
I adjusted the parameters in the following order:

*learning_rate
*dropout
*mu
*sigma
*batch_size
*test_valid_size

I initially created an architecture which yielded low prediction accuracy (~ 80.00%). I then choose the LeNet-5 architecture to help train the model.I choose this architecture because LeNet-5 is a convolutional neural network that is invariant to object translation
The architecture yielded optimal results.

My final model results were:
* validation set accuracy of 93.242630%
* test set accuracy of 92.232779%


**5. Test a Model on New Images**


Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]



The code for making predictions on my final model is located in the 12th code cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (30km/h)  | Speed limit (30km/h)	  						|
| Pedestrians    		| Pedestrians 									|
| Stop					| Stop										    |
| Beware of Ice/snow    | Beware of Ice/snow 					 		|
| Speed limit (60km/h)  | Speed limit (60km/h)      					|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.23%



**6. Analyze the softmax probabilities of the New Images**


The code for outputting the top 5 Softmax probabilities is located in the 14th code cell of the Ipython notebook.

The model was relatively sure of its prediction (probability of 0.9 and above). The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.48258221e-01        | Speed limit (30km/h)	   						|
| 5.17391078e-02    	| Speed limit (20km/h) 							|
| 2.60740262e-06		| Stop									        |
| 9.79431078e-08	    | Speed limit (120km/h)					 		|
| 8.55253108e-12	    | Speed limit (50km/h)	     					|

![alt text][image9]



For the second image ...


| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000000e+00        | Pedestrians   								|
| 1.93778438e-14     	| Road narrows on the right	                    |
| 1.30756732e-17		| Traffic signals						        |
| 5.37362003e-18	  	| Right-of-way at the next intersection			|
| 4.88287280e-18	    | Children crossing      						|


![alt text][image10]


For the third image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.70794737e-01        | Stop   									    |
| 2.91580372e-02     	| Speed limit (70km/h) 							|
| 4.69866609e-05		| Speed limit (120km/h)							|
| 2.61392699e-07	    | Keep left         					 		|
| 2.72137282e-08		| Speed limit (50km/h)    					    |


![alt text][image11]


For the fourth image ...

| Probability         	|     Prediction	        					      |
|:---------------------:|:---------------------------------------------------:|
| 1.00000000e+00        | Beware of Ice/snow   							      |
| 5.47114177e-12     	| Priority road     							      |
| 3.57251988e-12		| Roundabout mandatory              		          |
| 6.59795569e-13	    | Right-of-way at the next intersection			      |
| 3.75767024e-15	    | End of no passing by vehicles over 3.5 metrics tons |

![alt text][image12]


For the fifth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.99684334e-01        | Speed limit (60km/h)   						|
| 3.15544225e-04     	| Speed limit (80km/h) 							|
| 1.56212948e-07	    | Speed limit (120km/h) 						|
| 1.73570658e-09	    | Speed limit (20km/h) 					 		|
| 9.47238381e-13	    | Speed limit (30km/h)       					|

![alt text][image13]



**7. Visualize layer of the neural network**

Visualization of a layer in the neural network is done in the 15th code cell of the Ipython notebook. I used it to make sure the trained model learned important features of the various traffic signs.
Below are visualizations for the first and second layer in the neural network:


*Layer1*


![alt text][image20]



*Layer2*


![alt text][image21]

**8. Summary**

The dataset used to train the traffic sign classifier was loaded and preprocessed by converting to grayscale , resizing to 32 x 32 x 1 and adding augmented data. The model architecture used to train the traffic sign classifier is the LeNet-5 architecture .
The architecture yielded a validation accuracy of 93.24 and a test accuracy of 92.23%. Further , 5 random traffic sign images were selected form the web and tested against the traffic sign classifier. The traffic sign classifier correctly predicted all the images with high confidence. Visualization of the layers in the neural network served as a guide for tuning parameters while training the traffic sign classifier.


