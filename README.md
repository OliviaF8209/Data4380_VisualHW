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

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.






