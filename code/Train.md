# Train and evaluate (train.py)

- HINT: [Tensorflow](https://www.tensorflow.org/tutorials/keras/classification), [Pytorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

- Import all the necessary libraries, your model, loss function, metric, and dataset that you have implemented.

```python
# example
import tensorflow as tf
from src.data.dataloader import get_dataloader
from src.models.model import get_classification_model
from src.loss_function import binary_cross_entropy_loss_tf
from src.metric import accuracy_metric_tf

# get the dataset
train_dataloader, val_dataloader = get_dataloader()

# get the model
model = get_classification_model(input_shape=(28, 28), num_classes=10)

# get the loss function and metric
loss_function = binary_cross_entropy_loss_tf
metric = accuracy_metric_tf

for epoch in range(10):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    # training
    print("training...")
    train_loss = 0.0
    train_metric = 0.0
    for (images, labels) in enumerate(train_dataloader):
        # This is the derivative of the loss function with respect to the model's parameters in tensorflow
        # You may see the model.fit in tensorflow tutorial
        # But this is more flexible and you can use it in any model
        # The key word is "GradientTape"
        with tf.GradientTape() as tape:
            # Pass the input through the model
            predictions = model(images)
            # Calculate the loss
            loss = loss_function(labels, predictions)
            # Derivative of the loss with respect to the model's parameters
            gradients = tape.gradient(loss, model.trainable_variables)
            # Update the model's parameters
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # calculate the metric
            accuracy = metric(labels, predictions)

            # print the loss and metric
            train_loss += loss
            train_metric += metric
    train_loss /= len(train_dataloader)
    train_metric /= len(train_dataloader)
    print(f"train loss: {train_loss:.4f}")
    print(f"train metric: {train_metric:.4f}")

    # evaluation
    print("evaluating...")
    val_loss = 0.0
    val_metric = 0.0
    for batch, (images, labels) in enumerate(val_dataloader):
        predictions = model(images)
        loss = loss_function(labels, predictions)
        # evaluate does not need to calculate the gradient
        val_loss += loss
        val_metric += metric(labels, predictions)
    val_loss /= len(val_dataloader)
    val_metric /= len(val_dataloader)
    print(f"val loss: {val_loss:.4f}")
    print(f"val metric: {val_metric:.4f}")

# save the model
model.save_weights(f"model_{epoch}.h5")

```
