# Data preparation (dataloader.py)

- The main goal of this task is create a Dataloader class that can read the dataset and return the batch of data which can be used to train the model.

## 1. Download the dataset

- Download the dataset and store it in the dataset folder.
- Inspection the dataset and make sure that you know the structure of the dataset.
```bash
# example
# This is the structure of cat, dog dataset:
----dataset_folder
    ----train
        ----cat
            ----cat1.jpg
            ----cat2.jpg
        ----dog
            ----dog1.jpg
            ----dog2.jpg
    ----val
        ----cat
            ----cat1.jpg
            ----cat2.jpg
        ----dog
            ----dog1.jpg
            ----dog2.jpg
    ----test
        ----cat
            ----cat1.jpg
            ----cat2.jpg
        ----dog
            ----dog1.jpg
            ----dog2.jpg
# The dataset is split into 3 parts: train, val, test
# Each part has 2 classes: cat, dog
# The folder name is the class name
# The file in the folder is the image of the class
```

## 2. Preprocess the dataset

- Create a function that can read the image from the dataset folder and return the image and the label of the image. 
<br>&nbsp;&nbsp;&nbsp; Hint: use OpenCV or other libraries to read the image. (be careful with the format of the image, RGB or BGR depends on the library you use.)
- Create a function that do some preprocessing on the image.
<br>&nbsp;&nbsp;&nbsp; Resize the image to the same size. (input size of the model)
<br>&nbsp;&nbsp;&nbsp; Convert the image to the same format. (RGB, BGR, GRAY?)
<br>&nbsp;&nbsp;&nbsp; Convert the image to the same type. (float, int, etc)
<br>&nbsp;&nbsp;&nbsp; Normalize the image. (0-255 to 0-1, etc)
Hint: The above steps can be done by using OpenCV, numpy, default operators, etc.
- Preprocess the label if needed.
<br>&nbsp;&nbsp;&nbsp; Map the class name to the number. (cat: 0, dog: 1)
<br>&nbsp;&nbsp;&nbsp; Convert the label to the same format. (int, float, etc)
<br>&nbsp;&nbsp;&nbsp; For the classification problem, based on the loss function, you need to convert the label to the same format. For example, if you use the cross entropy loss, you need to convert the label to one-hot encoding.

## 3. Create the dataset
- Suppose you have:
```bash
read_image(image_path) # read the image from the dataset folder
preprocess_image(image) # preprocess the image
preprocess_label(label) # preprocess the label
```
- Create a class that inherits build-in dataset to create an dataloader for training model
<br>&nbsp;&nbsp;&nbsp; Hint: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence, https://pytorch.org/docs/stable/data.html
<br>&nbsp;&nbsp;&nbsp; Your dataloader should output the image and the label by batch.

```python
# example
import numpy as np
class Dataloader(YOUR_INHERIT_CLASS):
    def __init__(self, image_paths, labels, batch_size):
        # This is the constructor of the class
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        # This function should return the number of batch
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        image_path = self.image_paths[start_index:end_index]
        label = self.labels[start_index:end_index]
        images = []
        labels = []
        for i in range(len(image_path)):
            image = read_image(image_path[i])
            image = preprocess_image(image)
            label[i] = preprocess_label(label[i])
            images.append(image)
            labels.append(label[i])
        
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
dataset = Dataloader(image_paths, labels, batch_size)

dataiter = iter(dataset)
batch_images, batch_labels = next(dataiter)

# For the classification problem 
# The batch_images should be in the shape of (batch_size, height, width, channel) ex: (32, 224, 224, 3)
# The batch_labels should be in the shape of (batch_size, number_of_classes) ex: (32, 2) or (batch_size, 1) for non-one-hot encoding
```

Here your model only need the 1 input and 1 output. For some model, you may need more than 1 input and 1 output. For example, in the object detection problem, you need 1 input: image. And you need 2 output: the predicted bounding box and the predicted class. In this case, you need to create a class that inherits build-in dataset to create an dataloader for training model. The dataloader should output the image, the predicted bounding box, and the predicted class by batch.