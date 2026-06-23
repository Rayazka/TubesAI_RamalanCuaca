import math

class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implemented from scratch.
    """
    def __init__(self, k=5):
        """
        Initializes the classifier with the number of neighbors.
        
        Args:
            k (int): Number of neighbors to use for voting.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Fits the model by storing the training features and labels.
        
        Args:
            X (list of lists): Training features.
            y (list): Training labels.
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance_sq(self, x1, x2):
        """
        Computes the squared Euclidean distance between two vectors.
        We omit the square root operation because square root is a monotonic 
        transformation, meaning:
           if dist_sq(a, b) < dist_sq(a, c) then dist(a, b) < dist(a, c)
        Skipping square root saves significant CPU time in pure Python.
        
        Args:
            x1 (list): First vector.
            x2 (list): Second vector.
            
        Returns:
            float: Squared Euclidean distance.
        """
        dist_sq = 0.0
        # Accessing by index in a simple loop is faster in pure Python 
        # than using list comprehensions or zip()
        for i in range(len(x1)):
            diff = x1[i] - x2[i]
            dist_sq += diff * diff
        return dist_sq

    def predict_one(self, x):
        """
        Predicts the label for a single feature vector.
        
        Args:
            x (list): Feature vector.
            
        Returns:
            int: Predicted class (0 or 1).
        """
        # Calculate squared distance from x to all training points
        distances = []
        for i in range(len(self.X_train)):
            d = self._euclidean_distance_sq(x, self.X_train[i])
            distances.append((d, self.y_train[i]))
            
        # Sort distances in ascending order
        # Python's built-in Timsort is highly optimized
        distances.sort(key=lambda item: item[0])
        
        # Select the K nearest neighbors
        k_neighbors = distances[:self.k]
        
        # Count votes for each label
        votes = {}
        for _, label in k_neighbors:
            votes[label] = votes.get(label, 0) + 1
            
        # Find the label with maximum votes
        max_vote = max(votes.values())
        candidates = [label for label, count in votes.items() if count == max_vote]
        
        # If there's no tie, return the winner
        if len(candidates) == 1:
            return candidates[0]
            
        # Tie-breaker: Return the label of the closest neighbor among the tied candidates
        for _, label in k_neighbors:
            if label in candidates:
                return label
                
        return candidates[0] # Fallback

    def predict(self, X):
        """
        Predicts labels for a matrix of feature vectors.
        
        Args:
            X (list of lists): Feature matrix.
            
        Returns:
            list: List of predicted labels (0 or 1).
        """
        predictions = []
        for row in X:
            pred = self.predict_one(row)
            predictions.append(pred)
        return predictions
