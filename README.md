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

* Describe any manipulations you performed to the data.

#### Data Visualization

Show a few visualization of the data and say a few words about what you see.

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

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






