![](UTA-DataScience-Logo.png)

# Project Title

* **One Sentence Summary** Fruits-360 dataset is full of different types of fruits (Link: https://www.kaggle.com/datasets/moltean/fruits ). 

## Overview

* This section could contain a short paragraph which include the following:
  * **Definition of the tasks / challenge**  The challenge in this task is to classify fruit images into different categories using a dataset of fruit images. The goal is to use transfer learning with pre-trained models to build a classifier that can accurately categorize images into specified classes.
  * **Your approach** Ex: The approach in this repository utilizes transfer learning with pre-trained Keras vision models to address the image classification challenge> I also used transfer learning with pre-trained MobileNetV2 models to classify fruit images. The plan was to evaluate model performance using ROC curves, but I encountered errors during this process.
  * **Summary of the performance achieved** The performance of the models was planned to be assessed using ROC curves and AUC scores. The augmented model was expected to show improvements over the models trained without augmentation, highlighting the potential benefits of data augmentation in enhancing classification accuracy.
## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Type: Visual/Image
    * Input: Fruit/Vegetable images (100x100 pixel jpegs), CSV file: image filename -> diagnosis
  * Size: 94110 images
  * Instances (Train, Test, Validation Split): Training set size: 70491 images (one object per image), Test set size: 23619 images (one object per image), The number of classes: 141 (fruits and vegetables).

#### Preprocessing / Clean up

* - **Data Loading:** Images were loaded from directories using TensorFlow’s `image_dataset_from_directory`, ensuring images were resized to 224x224 pixels to match the input size of the pre-trained models.
  - **Data Augmentation:** Applied data augmentation techniques such as random horizontal flips and rotations to increase the variability of the training data and improve model generalization.
  - **Data Organization:** Cleaned up empty directories and ensured that only non-empty directories were used for training and testing.

#### Data Visualization

**Sample Images:** A few random images from each class were visualized to check data variety and integrity.

![image](https://github.com/user-attachments/assets/4f52735a-5b9c-41f0-bcea-de67d7d363b3)
![image](https://github.com/user-attachments/assets/637d2aef-140d-46b1-a6df-15ad5b6770a1)

**Augmented Images:** Displayed augmented images showing how techniques like flipping and rotation modify images to enhance model robustness.

![image](https://github.com/user-attachments/assets/384500d8-9951-4498-8312-50b728370fa9)
![image](https://github.com/user-attachments/assets/bd35aeae-9cc1-434e-b85e-29677b3ee83f)



### Problem Formulation

* **Define:**
  * **Input / Output:**
    - **Input:** Images of fruits resized to 224x224 pixels.
    - **Output:** Class labels corresponding to the type of fruit in the images (e.g., "Apple Red 2", "Blueberry 1").

  * **Models:**
    - **Baseline Model:** A transfer learning model using MobileNetV2 with its pre-trained weights, fine-tuned for fruit classification.
    - **Augmented Model:** The same MobileNetV2 model, but trained with additional data augmentation to improve generalization.
    - **Additional Models:** Two more models were trained with variations in architecture or parameters to compare performance.
   
Model1
![image](https://github.com/user-attachments/assets/046e9c92-b51e-4f81-9cec-fada11f29a88)

MOdel2
![image](https://github.com/user-attachments/assets/b7ae13b3-7150-468d-b997-f0ad2963fc40)

Augmented Model
![image](https://github.com/user-attachments/assets/67767a06-18fe-4771-abd3-557e551f975c)

  * **Loss, Optimizer, Other Hyperparameters:**
    - **Loss Function:** Sparse categorical cross-entropy, appropriate for multi-class classification.
    - **Optimizer:** Adam optimizer was used for its efficiency in training deep learning models.
    - **Other Hyperparameters:** Batch size was set to 30, and the number of epochs was 10 for initial training. Dropout was used to prevent overfitting.


### Training

* **Describe the training:**
  * **How You Trained:**
    - **Software:** Used TensorFlow and Keras for model implementation and training.
    - **Hardware:** Training was performed on a local machine with a GPU to accelerate the process.

  * **How Did Training Take:**
    - **Duration:** Each model was trained for 10 epochs, with training times varying based on model complexity and hardware performance.

  * **Training Curves (Loss vs. Epoch for Test/Train):**
    - **Plot:** Loss curves were plotted to visualize training and validation losses over epochs. This helped in monitoring overfitting and assessing model performance during training.

  * **How Did You Decide to Stop Training:**
    - **Stopping Criteria:** Training was stopped after 10 epochs or earlier if overfitting was detected or if the model's performance on validation data plateaued.

  * **Any Difficulties? How Did You Resolve Them?**
    - **Difficulties:** Encountered issues with data directory paths and dataset loading errors. Resolved by ensuring that all required directories were correctly populated and properly referenced in the code. Additionally, I tried to fix issues with model predictions and ROC curve plotting by adjusting dataset preparation, but the error persisted, claiming the input was empty. Even after confirming the input wasn't empty and removing empty directories, the problem remained unsolved.


### Performance Comparison

* **Key Performance Metrics:** The key metric intended for evaluation was the AUC score from ROC curves. Due to persistent errors with ROC curve plotting, this metric could not be computed. 

* **What Would Have Been Done:** If the errors were resolved, I would have compared models using ROC curves and AUC scores to determine which model performed best in distinguishing between classes.

### Conclusions

Data augmentation appeared to improve model performance based on training and validation loss curves, suggesting it enhances model robustness.

### Future Work

* **Next Steps:** Focus on resolving the errors with ROC curve plotting to accurately assess model performance using AUC scores.

* **Further Studies:** Explore additional performance metrics and refine dataset preprocessing to improve model evaluation and generalization.

## How to reproduce results

Applying to Other Data: To apply the trained model to another dataset, ensure the input data format matches the expectations (avoid errors) and adjust the model's output layers if necessary.

Resources: Using Google Colab with GPU acceleration is recommended for training the models efficiently.

### Overview of files in repository

File and Their Roles:

reduced_fruits_360/: Contains the dataset organized into Test and Train directories with subdirectories for each fruit class.
preprocessing section : Performs preprocessing steps on the dataset, including data cleaning and preparation.
training-models section : Contains code to define, train, and save three models with different configurations.
performance-evaluation section: (Loads trained models, evaluates their performance, and compares results.)

### Software Setup
Required Packages:
import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

TensorFlow
numpy
matplotlib
Installation Instructions:

Install TensorFlow and other packages using pip:
pip install tensorflow numpy matplotlib

## Data

**Download Location:**

Data is located in Data 4380/Image Project HW/reduced_fruits_360.

**Preprocessing Steps:**

Organize the dataset into Train and Test directories.
Ensure images are correctly labeled in their respective class directories.

### Training

**Training Instructions:**

Run training-models section to define, train, and save models.
Models are trained on images of size (100, 100, 3) and saved as model1.keras, model2.keras, and augmented_fruits_model.keras.


#### Performance Evaluation

**Evaluation Instructions:**
Use performance-evaluation.ipynb to load trained models and evaluate their performance.
Compare model performance using metrics such as accuracy and loss.


## Citations

* Fruits dataset link: https://www.kaggle.com/datasets/moltean/fruits
* Black Box AI: https://www.blackbox.ai/




