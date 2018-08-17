# 11. DNN Terminology


## Classification vs. Regression
https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/

Fundamentally, classification is about predicting a label and regression is about predicting a quantity.

## Softmax Function (i.e. Normalized Exponential Function)
https://en.wikipedia.org/wiki/Softmax_function

A generalization of the logistic function that "squashes" a K-dimensional vector ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/82eca5d0928078d5a61b9e7e98cc73db31070909) of arbitrary real values to a K-dimensional vector ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/2e610a6185b8850a6f567c4902387b17f0ec1652) of real values, where each entry is in the range (0, 1], and all the entries add up to 1.

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/86ead0d067436010ffe21c29fa4bf956eb023ff6)

![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/e348290cf48ddbb6e9a6ef4e39363568b67c09d3)

https://stats.stackexchange.com/questions/289369/log-probabilities-in-reference-to-softmax-classifier

Given a bunch of unnormalized log probabilities, and you want to recover the original probabilities, first you take the exponent of all your numbers, which gives you unnormalized probabilities. Next, you normalize them like usual.

https://medium.com/@uniqtech/understand-the-softmax-function-in-minutes-f3a59641e86d

![alt text](https://cdn-images-1.medium.com/max/1600/1*670CdxchunD-yAuUWdI7Bw.png)

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
## Cross Entropy Loss Function
https://sefiks.com/2017/12/17/a-gentle-introduction-to-cross-entropy-loss-function/

Cross entropy function correlates between probabilities and one hot encoded labels.

![alt text](https://i0.wp.com/sefiks.com/wp-content/uploads/2017/12/one-hot-encoding.png?resize=1140%2C608&ssl=1)
