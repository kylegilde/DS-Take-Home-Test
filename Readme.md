
# Part 1) Questions

## Data Science:

1. In the Least Angle Regression model (LASSO), what does the alpha parameter control?

In regularized regression, the alpha parameter controls the balance between the L1 and L2 regularizations.  If alpha is set to 1, this is known as a LASSO model. If it is set to 0, then this is a Ridge regression. If alpha is a value between 0 and 1, then this is an elastic net model, which mixes the effects of the LASSO and Ridge regressions.


2. Is more data always better? (Explain Why)

No, using more data to train and test a model (or for analysis) is not necessarily better. 

When training a model, it is imperative that the training data match the quality and nature of the testing data, and both datasets must be representative of the data that will be used in production. If any of these datasets are unaligned, then the model will not yield consistent results. 

Using more data to train and test a model can yield better results if it provides a superior representation of the data that will be used in production. However, if the new data is of better or worse in quality than the data used in production, then adding them will produce an unreliable model.

3. When do you use a Linear Discriminant Analysis (LDA) vs Principal Component Analysis (PCA)?

While both PCA and LDA are dimensionality reduction methods that look for linear combinations of features, PCA is suitable for either classification or regression models, but LDA can only be used with classification models.

4. What’s the difference between L1 and L2 regularization?

Both types of regularizations are used to create better models by preventing overfitting to the training data. The L1 penalty has the effect of shrinking coefficient values to zero, which effectively removes unimportant features from the model. On the other hand, the L2 penalty shrinks the size of the coefficient values to prevent multicollinearity.

5. Can a machine learning algorithm outperform a human? (Explain Why)

Yes, it is possible for an ML algorithm to have a smaller classification error rate than a human performing the same task. This occurred with image classification tasks and deep learning models. These models benefit from having been trained with data that was labeled by several people. Thus, the collective error rate is much smaller than the individual error rate.

## Statistics & Probability:

1. What is maximum likelihood estimation? Could there be any case where it doesn’t exist?

MLE is an approach to determining the values of the parameters for a model. It is the value of the parameter(s) that gives the largest probability to the observed data. An MLE may not exist if the likelihood function is not continuous in the parameter space and the parameter space is not compact.

2. Let C and D be two events with P(C) = 0.25, P(D) = 0.45, and P(C ∩ D) = 0.1. What is P(C^c ∩ D)? 

**0.75 * 0.45 = 0.3375**

3. Two dice are rolled.

A = ‘sum of two dice equals 3’ 

B = ‘sum of two dice equals 7’

C = ‘at least one of the dice shows a 1’

What is P(A|C)?  

**2/11 or 0.1818182**

## Deep Learning: 

1. How vanishing gradient problem can be solved?

One way to ameliorate this problem is to switch from using the sigmoid activation function to one that does not saturate as quickly, such as the rectified linear units function.

2. Explain batch gradient descent, stochastic gradient descent and mini-batch

The 3 types of gradient descent (GD) exhibit tradeoffs in "computational effeciency and the fidelity of the error gradient." The methods differ in the frequency of updates to the model and the quantity of the samples used to calculate the model error.

- Stochastic gradient descent (SGD) is often called online machine learning because it calculates the model error and updates the model for each new sample. Some of the possible advantages of SGD are faster learning, immediate insights into the performance and improvement rate of the model and avoiding premature convergence at a local minima. Some of the downsides are that it is computationally expensive, and the frequent updates can foster a noisy learning process.

- Batch gradient descent (BDG) differs from SGD in the frequency of updates to the model. BGD calcalutes the errors for all the samples, but model is only updated with batches of new samples after the completion of a training epoch. The advantages of BGD over SGD include that it is less computationally expensive and the learning process can be more stable. The disadvantages include a greater likelihood of premature convergence at suboptimal parameters, and model updates can be slow and require the entire dataset to fit in memory.

- Mini-batch GD is the most common GD technique used with deep learning models, and the one with which I have experience. It attempts to find a "balance between the robustness of stochastic gradient descent and the efficiency of batch gradient descent." In mini-batch GD, the model is still updated in batches, but it only calculates the model error on a subset of the samples instead of all the samples. This method is more computationally efficient than both BGD and SGD. The model updates are faster and one can update the model more frequently. The disadvantages include needing to tune mini-batch parameter and the needing multiple mini-batches to gain all the error information.

Source: [A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)

3. What is the difference between Transfer Learning and Fine-tuning?

- Transfer learning is applying a model that was initially trained for one task to different but related task. Starting with a pretrained model can save hours of training time for computer vision and natural language processing tasks.

- Fine-tuning refers to the process of making enhancements to the transferred model. These strategies include freezing, retraining or adding layers during additional epochs. 



