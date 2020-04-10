# Predicting iris variants

I want to gain intuition about how different supervised learning algorithms (logistic regression, neural network, support vector machine) perform on the same dataset. I have implemented these models and diagnostic tools to compare the results. The dataset I currently use is obtained from the UCI Machine Learning Repositary (http://archive.ics.uci.edu/ml/datasets/Iris) and contains data on the sepal and petal length and width of 3 iris variants. The task is to classify each flower sample as one variant of iris. I use the corrected dataset (refer to the UCI website). The dataset has 150 samples, 100 of which form the training set, 25 the cross-validation set and 25 the test set. There are 4 features (plus the constant/bias feature) and 3 groups.

## One-versus-all logistic regression classifier

The model fits 5 parameters (Theta) to the training set for each of the 3 groups. The regularisation parameter (Lambda) is fit to the cross-validation set (from the set {0, 0.01, 0.03, 0.1, 0.3, 1, 3}). The learning rate (Alpha) and the number of iterations (num_iters) hyperparameters for the batch gradient descent are chosen manually by observing the evolution of cost (for the whole sample, after the regularisation parameter had been set). The data is randomly shuffled at the beginning of each run (i.e. the 3 sample sets will be different), which introduces randomness in the output. This is why I talk about "consistent", "typical" and "average" output values.

Alpha is set to 0.1 (the gradient descent starts breaking down at Alpha=0.5) and num_iters is set to 2000; the algorithm runs fast (even on a laptop) with these values and the cost plot indicates that the gradient descent converges (there is not no improvement in cost or accuracy by setting num_iters=10000).

### Key takeaways

The algorithm consistently selects Lambda=0 based on the cross-validation set, so there is no reason to believe that the model overfits.

In the learning curve diagram, the training set cost converges very quickly to the cross-validation set cost (at a training set size of at most 30, this was consistent across different runs). This reinforces that there is no overfitting. Moreover, it is likely that the model underfits and including polinomial (or other functional form) terms in the features seems like a worthwhile effort to increase prediction accuracy.

The sample size of 150 is quite small and this causes a lot of variation in the accuracies. In 20 runs, the test set accuracy was between 0.88 and 1 with an average of 0.96 (one mistake out of 25 predictions on completely new data).

Iris-setosa is easy to distinguish from the other two variants while Iris-versicolor causes the most trouble. This is seen from an analysis of the errors of the predictor as well as substantially highers cost for the binary logistic classifier responsible for separating Iris-versicolor samples (elements 2000-4000 of cost_history) compared to the other two binary classifiers.

## Neural network
