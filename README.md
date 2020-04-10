# Logistic-regression-classifier

I implemented a one-versus-all logistic regression classifier to gain experience in coding a small machine learning project. I want to gain intuition about how different algorithms (logistic regression, neural network, support vector machine) perform on the same dataset. The dataset I currently use is obtained from the UCI Machine Learning Repositary (http://archive.ics.uci.edu/ml/datasets/Iris) and contains data on the sepal and petal length and width of 3 iris variants. The task is to classify each flower sample as one variant of iris. I use the corrected dataset (refer to the UCI website).

For the educational value, I coded the optimisation function (batch gradient descent) rather than using a more efficient Python optimisation package. I also implemented debugging features such as plotting the history of cost as the gradient descent iterates and learning curves. This allows me to play with the learning rate and see what's most efficient.
