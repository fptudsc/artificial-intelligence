# Inference

- Create a inference script which loads the model and runs inference on a given image. You need to make sure process your input the same way you did in the training script. You can use the following snippet as a reference.

```python
# example
import tensorflow as tf
from src.models.model import get_classification_model

# get the model
model = get_classification_model(input_shape=(28, 28), num_classes=10)

# load the model
model.load_weights("path/to/your/model")

# process the input
image = tf.image.resize(image, [28, 28])
image = image / 255.0
image = tf.expand_dims(image, axis=0)

# run inference
predictions = model(image)

# get the predicted label
predicted_label = tf.argmax(predictions, axis=1)

# print the predicted label
print(predicted_label)
```