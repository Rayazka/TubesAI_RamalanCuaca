import math

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier implemented from scratch.
    
    NOTE: This model is to be completed by Rayazka's partner.
    Below is the boilerplate code with guidelines and structure for integration.
    """
    def __init__(self):
        self.classes = []
        self.priors = {}       # Prior probabilities: {class_label: P(class)}
        self.means = {}        # Means per feature per class: {class_label: [mean_f1, mean_f2, ...]}
        self.variances = {}    # Variances per feature per class: {class_label: [var_f1, var_f2, ...]}
        self.is_fitted = False

    def fit(self, X, y):
        """
        Trains the Gaussian Naive Bayes model.
        Calculates class priors, feature means, and feature variances.
        
        Args:
            X (list of lists): Training features.
            y (list): Training labels.
        """
        # --- TODO: Partner's Implementation ---
        # 1. Identify unique classes (e.g., [0, 1]) and save to self.classes.
        # 2. Compute prior probability P(y) for each class and save to self.priors.
        # 3. Separate features in X by their class label in y.
        # 4. For each class, compute the mean and variance of each feature column:
        #    - Mean = sum(values) / count
        #    - Variance = sum((val - mean) ^ 2) / count
        #    - Store lists of means and variances in self.means and self.variances.
        # --------------------------------------
        
        # Temporary placeholder for integration testing (predicting all 0s)
        self.classes = list(set(y))
        self.priors = {c: y.count(c) / len(y) for c in self.classes}
        self.is_fitted = True
        print("[Naive Bayes] Boilerplate fit executed. (Waiting for actual implementation)")

    def _calculate_gaussian_pdf(self, x_val, mean, variance):
        """
        Calculates the probability density function (PDF) value for Gaussian distribution:
        P(x_i | y) = (1 / sqrt(2 * pi * var)) * exp( - (x_i - mean)^2 / (2 * var) )
        
        Args:
            x_val (float): Feature value.
            mean (float): Mean of the feature in the class.
            variance (float): Variance of the feature in the class.
            
        Returns:
            float: Probability density.
        """
        # --- TODO: Partner's Implementation ---
        # Hint: Add a small epsilon (e.g., 1e-9) to the variance to avoid division by zero!
        # --------------------------------------
        return 1.0

    def predict_one(self, x):
        """
        Predicts the class label for a single feature vector.
        Calculates the posterior probability for each class: P(y | X) proportional to P(y) * prod( P(x_i | y) )
        
        Args:
            x (list): Feature vector.
            
        Returns:
            int: Predicted class.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")
            
        # --- TODO: Partner's Implementation ---
        # Hint: Use logarithms to prevent underflow: 
        # log(P(y | X)) = log(P(y)) + sum( log(P(x_i | y)) )
        # Compare log posteriors for all classes and return the one with the highest value.
        # --------------------------------------
        
        return self.classes[0] # Temporary placeholder

    def predict(self, X):
        """
        Predicts class labels for a matrix of feature vectors.
        
        Args:
            X (list of lists): Feature matrix.
            
        Returns:
            list: List of predicted labels.
        """
        return [self.predict_one(row) for row in X]
