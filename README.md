# Predicting iris variants

I want to gain intuition about how different supervised learning algorithms (logistic regression, neural network, support vector machine) perform on the same dataset. I have implemented these models and diagnostic tools to compare the results. The dataset I currently use is obtained from the UCI Machine Learning Repositary (http://archive.ics.uci.edu/ml/datasets/Iris) and contains data on the sepal and petal length and width of 3 iris variants. The task is to classify each flower sample as one variant of iris. I use the corrected dataset (refer to the UCI website). The dataset has 150 samples, 100 of which form the training set, 25 the cross-validation set and 25 the test set. There are 4 features (plus the constant/bias feature) and 3 groups.

## One-versus-all logistic regression classifier

The model fits 5 parameters (Theta) to the training set for each of the 3 groups. The regularisation parameter (Lambda) is fit to the cross-validation set (from the set {0, 0.01, 0.03, 0.1, 0.3, 1, 3}). The learning rate (Alpha) and the number of iterations (num_iters) hyperparameters for the batch gradient descent are chosen manually by observing the evolution of cost (for the whole sample, after the regularisation parameter had been set). The data is randomly shuffled at the beginning of each run (i.e. the 3 sample sets will be different), which introduces randomness in the output. This is why I talk about "consistent", "typical" and "average" output values.

Alpha is set to 0.1 (the gradient descent starts breaking down at Alpha=0.5) and num_iters is set to 2000; the algorithm runs fast (even on a laptop) with these values and the cost plot indicates that the gradient descent converges (there is not no improvement in cost or accuracy by setting num_iters=10000).

### Key takeaways

The algorithm consistently selects Lambda=0 based on the cross-validation set, so there is no reason to believe that the model overfits.

In the learning curve diagram, the training set cost converges very quickly to the cross-validation set cost (at a training set size of at most 30, this was consistent across different runs). This reinforces that there is no overfitting. Moreover, it is likely that the model underfits and including polinomial (or other functional form) terms in the features seems like a worthwhile effort to increase prediction accuracy.

The sample size of 150 is quite small and this causes a lot of variation in the accuracies. In 20 runs, the test set accuracy was between 0.88 and 1 with an average of 0.96 (one mistake out of 25 predictions on completely new data).

Iris-setosa is easy to distinguish from the other two variants while Iris-versicolor causes the most trouble. This is seen from an analysis of the errors of the predictor as well as substantially higher cost for the binary logistic classifier responsible for separating Iris-versicolor samples (elements 2000-4000 of cost_history) compared to the other two binary classifiers. Looking at the Iris_dataset_scatterplot.svg file (By Nicoguaro - Own work, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=46257808), it is clear why this is the case: the setosa variant is set apart distinctly from the other two variants; the classifier never makes a mistake on a setosa sample. The versicolor and virginica variants are much closer and overlap in some 2D subspaces of the 4D feature space; furthermore the versicolor samples tend to be closer to setosa samples. Intuitively, the classifier can find a linear decision boundary that separates setosa samples from the rest easily; a linear decision boundary still does well at separating virginica samples, although (with a very small probability) a virginical will be confused for a versicolor); but there is no linear decision boundary to separate versicolor well because its samples lie inbetween samples of the other two variants in **ALL** 2D subspaces. This reinforces that we should add polinomial features, so that the decision boundary around versicolor samples can be an ellipsoid.

I committed 2 mistakes: visualised the data only after running the algorithm (so I had no idea which group could be problematic or that polinomial features will be useful for classifying versicolor samples) and I explicitly wrote the data dimensions in the algorithm instead of obtaining it from the sample (so running on other datasets or with extra features will require some adjustment).

## Neural network

I implemented a 1-hidden-layer neural network with 3 neurons (plus the constant neuron) in the hidden layer (so there are 3x5+3x4=27 weights in total). The neural network works for general sample size m, number of features n and number of groups K. The number of neurons in the hidden layer can also be changed by adjusting a variable. Hence the only restriction is that there is only one hidden layer; this is accepable in our case because, with a training set size of 100, we would get bad overfitting with additional layers.

Hyperparameters in the model are therefore the size of the hidden layer H, the regularisation parameter Lambda, the learning rate Alpha and the number of iterations for the optimisation algorithm (batch gradient descent). These, or least H and Alpha, should be chosen by the algorithm with respect to the cross-validation set, but I have not implemented this feature (see below).

### Key takeaways

The whole thing went **surprisingly** easily and worked neatly. The network architecture (H=3) was chosen arbitrarily but turned out to work, I have not (yet) experimented with changing it. I was worried that 27 parameters would overfit the training set but this did not turn out to be the case: even without regularisation, there was no noticable difference in the average performance on the training set versus the cross-validation set across multiple runs.



to write about: flat part of cost curve
optimisation algorithm
use pre-defined models
