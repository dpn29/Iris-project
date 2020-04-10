# Predicting iris variants

I want to gain intuition about how different supervised learning algorithms (logistic regression, neural network, support vector machine) perform on the same dataset. I have implemented these models and diagnostic tools to compare the results. The dataset I currently use is obtained from the UCI Machine Learning Repositary (http://archive.ics.uci.edu/ml/datasets/Iris) and contains data on the sepal and petal length and width of 3 iris variants. The task is to classify each flower sample as one variant of iris. I use the corrected dataset (refer to the UCI website). The dataset has 150 samples, 100 of which form the training set, 25 the cross-validation set and 25 the test set. There are 4 features (plus the constant/bias feature) and 3 groups.

## One-versus-all logistic regression classifier

The model fits 5 parameters (Theta) to the training set for each of the 3 groups. The regularisation parameter (Lambda) is fit to the cross-validation set (from the set {0, 0.01, 0.03, 0.1, 0.3, 1, 3}). The learning rate (Alpha) and the number of iterations (num_iters) hyperparameters for the batch gradient descent are chosen manually by observing the evolution of cost (for the whole sample, after the regularisation parameter had been set). The data is randomly shuffled at the beginning of each run (i.e. the 3 sample sets will be different), which introduces randomness in the output. This is why I talk about "consistent", "typical" and "average" output values.

### Key diagnostics

The algorithm consistently selects Lambda=0 based on the cross-validation set, so there is not reason to believe that the model overfits.
