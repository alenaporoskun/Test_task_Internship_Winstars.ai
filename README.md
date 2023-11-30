# Test_task_Internship_Winstars.ai

R&D center WINSTARS.AI
Test task in Data Science  

During working in the company you will be faced with different computer vision tasks. So it is useful to have skills in image segmentation. After completing this test task you will be able to implement similar algorithms in commercial projects. 
Current task requires solving one of the problems from the Kaggle platform: [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview).

You can find the dataset here: dataset for test task [dataset for test task](https://www.kaggle.com/c/airbus-ship-detection/data).

The goal of the test project is to build a semantic segmentation model. Prefered tools and notes: tf.keras, Unet architecture for neural network, dice score, python. 

Results of your work should contain next:
*	link to GitHub repository with all source codes;
*	code for model training and model inference should be separated into different .py files;
*	readme.md file with complete description of solution;
*	requirements.txt with required python modules;
*	jupyter notebook with exploratory data analysis of the dataset;
*	any other things used during the working with task;
  
*	Source code should be well readable and commented;
*	Project has to be easy to deploy for testing;


**Training a model for image segmentation ([train.py](train.py)), particularly for detecting the presence of ships in images, involves the following steps:**

Data Preparation:
Loading training images that include ship images along with corresponding masks indicating the ship locations.
Splitting the data into training and validation sets.

Building a U-Net Model for Segmentation:
Defining the architecture of the U-Net model capable of reproducing images and capturing spatial information.
Adding an input layer, encoder, a multi-level bottleneck, and a decoder.
Using the sigmoid activation function in the final layer to obtain segmentation masks.

Compiling the Model:
Choosing a loss function (e.g., binary cross-entropy) and an optimizer (e.g., Adam).
Selecting metrics for evaluating segmentation accuracy.

Training the Model:
Using training images and corresponding masks to improve the model parameters.
Adjusting the number of epochs and other hyperparameters to achieve satisfactory results.

Model Evaluation:
Utilizing the validation set to assess segmentation accuracy and avoid overfitting.

Parameter Tuning:
Experimenting with hyperparameters, such as batch size, learning rate, and others, to optimize training.
  
Model Testing:
Employing a test set for the final evaluation of the model's performance on new images with ships.

Saving the Model:
Saving the trained model for future use in other scenarios or for making predictions on new data.

