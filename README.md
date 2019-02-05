## Project: Build a Traffic Sign Recognition Classifier

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

## Step 0: Load The Data set

    I used python and pickle library to load the data from 'train.p', 'valid.p' and 'test.p' pickle files
    Also stroring the data in the features and label format

## Step 1: Dataset Summary & Exploration


#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

    I used the python and numpy library to calculate summary statistics of the traffic sign data set:
    
    The size of training set is 34799
    The size of the validation set is 4410
    The size of test set is 12630
    The shape of a traffic sign image is (32, 32, 3)
    The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

    Here using csv library reading all the labels for signs. And displaying 1 example image for each category sign class with there label name.
    
    Here i am plotting the occurrence of each image class to get an idea of how the data is distributed for training , validation and testing data set.i am using numpy unique function to calculate each image occurrence.
    
## Step 2: Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

        I converted the images into grayscale and then used  (pixel - 128)/ 128 to normalized the data which provide mean zero and equal variance for image data. Which helps in speed of training and increase performance.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

        I used LeNet model shown in the classroom. Did few changes now the image has 1 color channel instead of 3. And added 43 classes in last layer.  
    
    using mu = 0, sigma = 0.1 which helps in initializing the weight.

    My final model consisted of the following layers:

| Layer         		|     Description	        					            | 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled normalized image   				    | 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	            |
| RELU					|												            |
| Max pooling	      	| 2x2 kernel, 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	            |
| RELU					|												            |
| Max pooling	      	| 2x2 kernel, 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten               | input 5x5x16, output 400                                                |
| Fully connected		| input 400, output 120									    |
| RELU					|												            |
| Dropout    			| 50 % keep during training, 100% keep during prediction    |
| Fully connected		| input 120, output 84									    |
| RELU					|												            |
| Dropout    			| 50 % keep during training, 100% keep during prediction    |
| Fully connected		| input 84, output 43									    |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

    To train the model with following details:
    
    epochs : 200
    batch size : 256
    learning rate : 0.00099
    optimizer : AdamOptimizer
    
    Previously i was augmented the data but there is no improvement in the validation accuracy. so i removed those steps.
    Flipping of the images is also not an good idea because if we flip the images the sign on the sign image is not correct.
 
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
    
    
My final model results were:
  * training set accuracy of 100.00 %
  * validation set accuracy of 96.90 %? 
  * test set accuracy of 94.80 %
  * test data accuracy from web of 87.50 % 

If an iterative approach was chosen:
  * What was the first architecture that was tried and why was it chosen?
      - I used the lenet architecture which was given as a good candidate for this kind of probblem.
      
  * What were some problems with the initial architecture?
      - Issue happend related to accurracy. Accuracy is good on training but not on validation. So got the problem of overfitting.
      
  * How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
     - So to resolve the probem of overfitting. First i augmented the data by flipping the images. Then accuracy got worse. After some time i realized flipping the images is not an good idea for this kind of problem. Because if we flip the image the meaning of the sign got changes. So i removed the augmentation and added the Dropout with 50% keep probability. I also changed RGB image into grayscale and normalized the image which will help to speed the training process and increase in improvement.
     
  * Which parameters were tuned? How were they adjusted and why?
     - Different combination of Epoch, learning rate, batch size, and dropout probability were used to get optimal accurracy. After a lot of parameter changes i choose epochs 200, learning rate 0.00099, batch size 256 and dropout probability 50 % which gives me 100 % training and 96.90 % validation accuracy. I also checked on test data accuracy given for project which is 94.80 %.
     
     
## Step 3: Test a Model on New Images


#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
    
     I downloaded 8 images from the web. Below is those 8 test images.
     
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
    
    The model was able to correctly guess 7 out of 8 traffic signs images, which gives an accuracy of 87.50%. Only 1 image is recognize wrong.

Here are the results of the prediction:

|   Actual Label(Index)        |   Predicted Label(Index)      | 
|:----------------------------:|:-----------------------------:| 
| Children crossing (28)       | Children crossing (28)        |
| General caution (18)         | General caution (18)          |
| No entry (17)                | No entry (17)                 |
| Road work (25)               | Road work (25)                |
| Speed limit (70km/h) (4)     | Speed limit (70km/h) (4)      |
| Stop (14)                    | Stop (14)                     |
| Turn right ahead (33)        | Turn right ahead (33)         |
| Wild animals crossing (31)   | Slippery road (23)            |


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

    The top five soft max probabilities for all the downloaded 8 images are as follows:

|    Actual Image (Index)     |         Softmax Probability                       |
|:---------------------------:|:-------------------------------------------------:|
|  Children crossing (28)     |   Children crossing: 62.01%                       |
|                             |   Beware of ice/snow: 37.99%                      |
|                             |   Right-of-way at the next intersection: 0.00%    |
|                             |   Turn left ahead: 0.00%                          |
|                             |   Pedestrians: 0.00%                              |
|  General caution (18)       |   General caution: 100.00%                        |
|                             |   Pedestrians: 0.00%                              |
|                             |   Speed limit (20km/h): 0.00%                     |
|                             |   Speed limit (30km/h): 0.00%                     |
|                             |   Speed limit (50km/h): 0.00%                     |
|  No entry (17)              |   No entry: 100.00%                               |
|                             |   Stop: 0.00%                                     |
|                             |   Speed limit (30km/h): 0.00%                     |
|                             |   Turn right ahead: 0.00%                         |
|                             |   Keep left: 0.00%                                |
|  Road work (25)             |   Road work: 100.00%                              |
|                             |   General caution: 0.00%                          |
|                             |   Ahead only: 0.00%                               |
|                             |   Dangerous curve to the right: 0.00%             |
|                             |   Double curve: 0.00%                             |
|  Speed limit (70km/h) (4)   |   Speed limit (70km/h): 100.00%                   |
|                             |   Speed limit (30km/h): 0.00%                     |
|                             |   Speed limit (20km/h): 0.00%                     |
|                             |   Speed limit (50km/h): 0.00%                     |
|                             |   Speed limit (60km/h): 0.00%                     |
|  Stop (14)                  |   Stop: 100.00%                                   |
|                             |   No entry: 0.00%                                 |
|                             |   Turn right ahead: 0.00%                         |
|                             |   Speed limit (30km/h): 0.00%                     |
|                             |   Yield: 0.00%                                    |
|  Turn right ahead (33)      |   Turn right ahead: 100.00%                       |
|                             |   Yield: 0.00%                                    |
|                             |   Keep left: 0.00%                                |
|                             |   Speed limit (50km/h): 0.00%                     |
|                             |   Ahead only: 0.00%                               |
|  Wild animals crossing (31) |   Slippery road: 100.00%                          |
|                             |   Wild animals crossing: 0.00%                    |
|                             |   Dangerous curve to the right: 0.00%             |
|                             |   Speed limit (60km/h): 0.00%                     |
|                             |   Speed limit (20km/h): 0.00%                     |

## Conclusion:

1 - I think for few image there is small number of example so to get better result i will improve my data preprocessing step where i will augment the data in future.