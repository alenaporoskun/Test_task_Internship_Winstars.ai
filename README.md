# Test_task_Internship_Winstars.ai

R&D center WINSTARS.AI
Test task in Data Science  

During working in the company you will be faced with different computer vision tasks. So it is useful to have skills in image segmentation. After completing this test task you will be able to implement similar algorithms in commercial projects. 
Current task requires solving one of the problems from the Kaggle platform: [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/overview).

You can find the dataset here: dataset for test task [dataset for test task](https://www.kaggle.com/c/airbus-ship-detection/data).

The goal of the test project is to build a semantic segmentation model. Prefered tools and notes: tf.keras, Unet architecture for neural network, dice score, python. 

Results of your work should contain next:
*	link to GitHub repository with all source codes;
*	code for model training and model inference should be separated into different .py files ([train.py](train.py), [test.py](test.py));
*	readme.md file with complete description of solution ([README.md](README.md));
*	requirements.txt with required python modules ([requirements.txt](requirements.txt));
*	jupyter notebook with exploratory data analysis of the dataset ([Exploratory_data_analysis.ipynb](Exploratory_data_analysis.ipynb));
*	any other things used during the working with task;
  
*	Source code should be well readable and commented;
*	Project has to be easy to deploy for testing;


## Training ([train.py](train.py))

**Training a model for image segmentation, particularly for detecting the presence of ships in images, involves the following steps:**
  
* Data Preparation:  
Loading training images that include ship images along with corresponding masks indicating the ship locations.
Splitting the data into training and validation sets.
  
* Building a U-Net Model for Segmentation:  
Defining the architecture of the U-Net model capable of reproducing images and capturing spatial information.
Adding an input layer, encoder, a multi-level bottleneck, and a decoder.
Using the sigmoid activation function in the final layer to obtain segmentation masks.
  
* Compiling the Model:  
Choosing a loss function (e.g., binary cross-entropy) and an optimizer (e.g., Adam).
Selecting metrics for evaluating segmentation accuracy.

* Training the Model:  
Using training images and corresponding masks to improve the model parameters.
Adjusting the number of epochs and other hyperparameters to achieve satisfactory results.

* Model Evaluation:  
Utilizing the validation set to assess segmentation accuracy and avoid overfitting.

* Parameter Tuning:  
Experimenting with hyperparameters, such as batch size, learning rate, and others, to optimize training.

* Saving the Model [/model](model):  
Saving the trained model for future use in other scenarios or for making predictions on new data.
  

## Testing ([test.py](test.py))

Results - [/result-image](result-image).

Testing a segmentation model involves evaluating its performance on a separate set of test images that were not used during training. Here are the general steps for testing a segmentation model:
  
* Load the Pre-trained Model:  
Load the saved model that was trained on the training data.
  
* Prepare Test Data:  
Collect a set of test images that the model has not seen before.  
If ground truth masks are available for the test images, they can be used for evaluation.
  
* Preprocess Test Images:  
Preprocess the test images in the same way as the training images (resize, normalization, etc.).
  
* Model Inference:  
Use the pre-trained model to make predictions on the test images.
  
* Post-process Predictions:  
If necessary, post-process the model predictions. This might involve thresholding, morphological operations, or other techniques depending on the specific segmentation task.  
  
* Evaluation Metrics:  
If ground truth masks are available for the test set, compute evaluation metrics such as Intersection over Union (IoU), Dice coefficient, accuracy, etc.  
Compare the model predictions with the ground truth to assess the quality of segmentation.
  
* Visualization:  
Visualize the test images along with the model predictions to qualitatively assess how well the model is performing.
  
* Iterative Improvement:  
If the model performance is not satisfactory, consider fine-tuning the model, adjusting hyperparameters, or collecting additional labeled data for retraining.
  
* Deployment:  
Once satisfied with the model's performance on the test set, the model can be deployed for making predictions on new, unseen data.  
The specific steps and metrics for testing may vary based on the segmentation task and the nature of the data.


## References
[https://www.kaggle.com/code/aditya100/airbus-ship-detection-unet](https://www.kaggle.com/code/aditya100/airbus-ship-detection-unet)
