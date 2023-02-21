# Loss function and metric (metris.py, loss_function.py)

- You need to understand the loss function and metric of the problem that you want to solve.

- This is an example of loss function and metric of binary classification problem:

```python
def binary_cross_entropy_loss_tf(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def accuracy_metric_tf(y_true, y_pred):
    y_pred = tf.cast(tf.greater(y_pred, 0), tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
```

P/s: You can use the built-in loss function and metric of tensorflow (keras) or pytorch.