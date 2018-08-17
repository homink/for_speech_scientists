# 11. DNN Terminology


## Classification vs. Regression
https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/

Fundamentally, classification is about predicting a label and regression is about predicting a quantity.

https://www.reddit.com/r/MachineLearning/comments/3klqdh/q_whats_the_difference_between_crossentropy_and/

In terms of loss function,  mean squared error is appropriate to regression (line/curve fitting) where the goal is to minimize the mean squared error between the training set (points) and the fitted curve. Cross entropy cost is appropriate to classification where the goal is to minimize the number of mis-classified training samples by imposing an exponentially increasing error the closer an output comes to being "1" when it should be "0", and vice versa.

## Softmax Function (i.e. Normalized Exponential Function)
https://en.wikipedia.org/wiki/Softmax_function

A generalization of the logistic function that "squashes" a K-dimensional vector ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/82eca5d0928078d5a61b9e7e98cc73db31070909) of arbitrary real values to a K-dimensional vector ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/2e610a6185b8850a6f567c4902387b17f0ec1652) of real values, where each entry is in the range (0, 1], and all the entries add up to 1.

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/86ead0d067436010ffe21c29fa4bf956eb023ff6)

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/e348290cf48ddbb6e9a6ef4e39363568b67c09d3)

https://stats.stackexchange.com/questions/289369/log-probabilities-in-reference-to-softmax-classifier

Given a bunch of unnormalized log probabilities, and you want to recover the original probabilities, first you take the exponent of all your numbers, which gives you unnormalized probabilities. Next, you normalize them like usual.

https://medium.com/@uniqtech/understand-the-softmax-function-in-minutes-f3a59641e86d

![alt text](https://cdn-images-1.medium.com/max/880/1*670CdxchunD-yAuUWdI7Bw.png?w=100)

```
logits = [2.0, 1.0, 0.1]
import numpy as np
exps = [np.exp(i) for i in logits]
sum_of_exps = sum(exps)
softmax = [j/sum_of_exps for j in exps]
>>> softmax
[0.6590011388859679, 0.2424329707047139, 0.09856589040931818]
>>> sum(softmax)
1.0
```
## Cross Entropy Loss Function (i.e. Averaged Cross-Entropy Loss Function and Log Loss Function)

https://medium.com/@dmitrijtichonov/debunking-loss-functions-in-deep-learning-4b9abc4c8d4c

![alt text](https://cdn-images-1.medium.com/max/880/1*AlbV9jz2k3Ll1wEMCljdSg.png?w=100)

## Regularization

Controlling the capacity of Neural Networks to prevent overfitting

### a. L1 and L2 regularization

![alt text](https://i2.wp.com/laid.delanover.com/wp-content/uploads/2018/01/reg_formulas.png?w=400 )

### b. Dropout

An extremely effective, simple and recently introduced regularization technique that keeps a neuron active with some probability p (a hyperparameter), or setting it to zero otherwise while training.

![](http://cs231n.github.io/assets/nn2/dropout.jpeg)

## Weight Initialization

### a. Batch normalization

![](https://cdn-images-1.medium.com/max/800/1*Hiq-rLFGDpESpr8QNsJ1jg.png)

### b. Xavier initialization

![](https://image.slidesharecdn.com/weightsinitialization-170118090134/95/weights-initialization-7-638.jpg?cb=1484730203)
