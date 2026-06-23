import math

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes Classifier implemented from scratch.
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
        n_samples = len(X)
        self.classes = list(set(y))
        
        # 1. Compute class priors: P(c) = count(c) / total_samples
        for c in self.classes:
            class_count = sum(1 for label in y if label == c)
            self.priors[c] = class_count / n_samples
            
        # Separate features in X by class
        X_by_class = {c: [] for c in self.classes}
        for features, label in zip(X, y):
            X_by_class[label].append(features)
            
        # 2. For each class, compute mean and variance for each feature column
        num_features = len(X[0])
        for c in self.classes:
            self.means[c] = []
            self.variances[c] = []
            class_samples = X_by_class[c]
            num_class_samples = len(class_samples)
            
            # Compute means
            for j in range(num_features):
                feature_vals = [sample[j] for sample in class_samples]
                mean_val = sum(feature_vals) / num_class_samples
                self.means[c].append(mean_val)
                
            # Compute variances
            for j in range(num_features):
                feature_vals = [sample[j] for sample in class_samples]
                mean_val = self.means[c][j]
                # Variance: sum((x - mean)^2) / count
                variance_val = sum((val - mean_val) ** 2 for val in feature_vals) / num_class_samples
                self.variances[c].append(variance_val)
                
        self.is_fitted = True
        print("[Naive Bayes] Model successfully trained from scratch.")

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
        # Add a small epsilon to avoid division by zero
        eps = 1e-9
        var = variance + eps
        
        # Calculate exponents
        exponent = math.exp(-((x_val - mean) ** 2) / (2 * var))
        
        # Calculate full PDF
        return (1.0 / math.sqrt(2.0 * math.pi * var)) * exponent

    def predict_one(self, x):
        """
        Predicts the class label for a single feature vector.
        Calculates the posterior probability for each class: 
        P(y | X) is proportional to P(y) * prod( P(x_i | y) )
        
        We use log sums to prevent floating point underflow:
        log(P(y | X)) = log(P(y)) + sum( log(P(x_i | y)) )
        
        Args:
            x (list): Feature vector.
            
        Returns:
            int: Predicted class.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")
            
        posteriors = {}
        for c in self.classes:
            # log( P(y) )
            log_prior = math.log(self.priors[c])
            
            # sum( log( P(x_i | y) ) )
            log_likelihood = 0.0
            for j in range(len(x)):
                mean = self.means[c][j]
                variance = self.variances[c][j]
                
                pdf_val = self._calculate_gaussian_pdf(x[j], mean, variance)
                
                # Use a small epsilon to prevent math.log(0) error if pdf_val is 0
                eps = 1e-15
                log_likelihood += math.log(pdf_val + eps)
                
            posteriors[c] = log_prior + log_likelihood
            
        # Return the class with the highest log posterior probability
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        """
        Predicts class labels for a matrix of feature vectors.
        
        Args:
            X (list of lists): Feature matrix.
            
        Returns:
            list: List of predicted labels.
        """
        return [self.predict_one(row) for row in X]
