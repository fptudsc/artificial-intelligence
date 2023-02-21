In this assignment, you will:
- Learn to overcome the OOM (out-of-memory) issue when the dataset size is too large by splitting dataset into **batch size**.
- Implement the **R-Squared** metrics for evaluating the model.

# Instructions
The main file you are working with in this assignment is at [PATH!](code/src/models/model.py)

In this file, you will need to complete every `TODO` by replacing the `None` with a proper code to make the algorithm works correctly.

This is more details and hints:
```
Remember to do it yourself before watching hints
```
## TODO-1: Shuffle dataset after each iteration (remember to map X and y correctly)
Shuffling the dataset is an important step in machine learning because it helps to remove any biases or patterns that may exist in the data. When data is collected or prepared, it may be ordered in a certain way, for example, by the time it was collected or by a certain category. If the data is not shuffled before it is split into training and validation sets, the training set may end up with a biased representation of the data, which can lead to poor performance on the validation set and ultimately poor model performance.

[Hint](https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html)

## TODO-2: Calculate the output, loss and update weights and bias
Remember to calculating correctly by the mean of the `batch_size` (not the whole dataset length)

## TODO-3: R-Squared metric
Take a look at [R-squared formula](https://www.google.com/search?q=r-squared+formula&source=lmns&bih=944&biw=1920&hl=en&sa=X&ved=2ahUKEwiGpcfLlaf9AhVJgFYBHXYUCz0Q_AUoAHoECAEQAA) and code it yourself!