# How to use this ProjectTemplate

## 1. Folder Structure
- These are 4 subfolders in the project folder
    - `code`: this is the folder storing all the project code
    - `dataset`: Please look at the `README.md` in this folder
    - `git-repo`: Please look at the `README.md` in this folder
    - `research`: Please look at the `README.md` in this folder

## 2. Workflow

### 2.1. Research

- This is an important step in the project. It is the step where you do the research and find the best way to solve the problem.

-> How to research:

---- 
- Step 1: Define the problem
<br>You need to clearly know what the problem is and what you want to achieve.
For examples:
```
"You have a request to build a model which can predict the cat or dog in the image."
```
<br> -> Question: What is the input? 
<br> -> Answer: The image of the cat or dog. 
<br> -> More specifically, the image of the cat or dog is a 3D array of sizes (height, width, channel).

<br> -> Question: What is the output? 
<br> -> Answer: The cat if the input is a cat image, the dog if the input is a dog image.
<br> -> More specifically, the output is a scalar of 0 or 1. 0 means the input is a cat image, 1 means the input is a dog image.

<br> -> Question: What is the type of the problem?
<br> -> Answer: The problem is a classification problem.
<br> -> More specifically, the problem is a binary classification problem.

If you don't know what the problem is, you can't solve it. So, you need to clearly define the problem first.

- Step 2: Find the best way to solve the problem based on step 1
<br>After you clearly know what the problem is, you need to find the method to solve the problem. Paperwithcode.com, scholar.google.com, kaggle.com, etc are good places to find the best way to solve the problem. Tips: you can use the keywords in step 1 to search.

- Step 2.1: What you need to look at when you find a method (deep learning) can solve the problem?
<br>What is the input? 
<br>What is the output? 
<br>What is the model architecture? 
<br>What is the loss function? 
<br>What is the optimizer? 
<br>What is the metric? 
<br>What are the hyperparameters?
<br>Is there any provided code? Try to use the provided code first and see if it works.

P/s: this is just a general guideline.

- Step 2.1: Dataset
<br> If you don't have the dataset, you need to find the dataset. You can find the dataset on kaggle.com, google.com, etc.
<br> The preprocessed method that you need to apply to your dataset.
<br> The format of the dataset (csv, json, folder, jpg, etc)
<br> The information of the dataset (number of samples, number of classes, etc), choose the best dataset for your problem.
<br> Train, val, test dataset

- Step 3: Plan the project
<br> After you find the best way to solve the problem, you need to plan the project. You need to know what you need to do in the project. For example:
<br> - Step 1: Download the dataset
<br> - Step 2: Preprocess the dataset
<br> - Step 3: Build the model
<br> - Step 4: Train the model
<br> - Step 5: Evaluate the model
<br> - Step 6: Deploy the model
<br> ***Or you only need to do some of the steps above. For example:***
<br> - Step 1: Preprocess the dataset
<br> - Step 2: Train the model (use some other's code)
<br> - Step 3: Evaluate the model

### 2.2. Dataset preparation

- Please look at [here](code/src/data/Readme.md)

### 2.3. Model implementation

- Please look at [here](code/src/models/Readme.md)

### 2.4 Loss function and metric

- Please look at [here](code/src/LossMetrics.md)

### 2.5. Train and evaluate

- Please look at [here](code/Train.md)

### 2.6 Inference

- Please look at [here](code/Inference.md)

### 2.7 Documentation

- Please create a Readme.md in the `code` folder. This is the documentation of the code which let other people know how to use your code. For example: how to create environment, how to run the code, how to train the model, how to evaluate the model, etc.

- You may want to provide a notebook to show how to use your code in notebooks folder, some images in docs folder, etc.

