# Model implementation

- The main goal of this task is to implement the model which receives the batch of data and return the prediction of the data.
- Hint: 
<br>TF:  https://www.tensorflow.org/tutorials/keras/classification#:~:text=model%20%3D%20tf.keras.Sequential(%5B%0A%C2%A0%20%C2%A0%20tf.keras.layers.Flatten(input_shape%3D(28%2C%2028))%2C%0A%C2%A0%20%C2%A0%20tf.keras.layers.Dense(128%2C%20activation%3D%27relu%27)%2C%0A%C2%A0%20%C2%A0%20tf.keras.layers.Dense(10)%0A%5D
<br>Pytorch: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#:~:text=import%20torch.nn,%3D%20Net()


P/s: These link is have a complete example of how to train a model. You can use it as a reference. Please make sure that you understand the code before you use it.

```python
# example
import tensorflow as tf
from tensorflow.keras import layers

def get_classification_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
```