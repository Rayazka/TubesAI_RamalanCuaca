import csv
import random

def load_data(file_path, target_threshold=3.0):
    """
    Reads the weather CSV file, separates features and target, and removes leakage columns.
    
    Args:
        file_path (str): Path to the CSV dataset.
        target_threshold (float): Threshold in mm to define rainy (1) vs dry (0).
        
    Returns:
        X (list of lists): Feature matrix.
        y (list): Target labels (0 or 1).
        feature_names (list): Names of the features used.
    """
    X = []
    y = []
    
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Clean column names (remove weird characters/spaces)
        header = [col.strip() for col in header]
        
        # Find column indexes
        try:
            target_idx = header.index("precipitation_sum (mm)")
        except ValueError:
            raise ValueError("Target column 'precipitation_sum (mm)' not found in dataset header.")
            
        # Define leakage/unused columns to drop
        # 'time' is date (string)
        # 'precipitation_sum (mm)' is target
        # 'precipitation_hours (h)' is direct leakage
        leakage_cols = ["time", "precipitation_sum (mm)", "precipitation_hours (h)"]
        drop_indices = set()
        for col_name in leakage_cols:
            if col_name in header:
                drop_indices.add(header.index(col_name))
                
        # Define the remaining feature indices and names
        feature_indices = [i for i in range(len(header)) if i not in drop_indices]
        feature_names = [header[i] for i in feature_indices]
        
        # Process rows
        for row in reader:
            if not row:
                continue
            try:
                # Target value
                precip = float(row[target_idx])
                label = 1 if precip > target_threshold else 0
                
                # Feature values
                features = [float(row[i]) for i in feature_indices]
                
                X.append(features)
                y.append(label)
            except ValueError as e:
                # Skip header repetitions or invalid rows (if any)
                continue
                
    return X, y, feature_names

def fit_min_max(X_train):
    """
    Calculates the min and max for each feature in the training set.
    
    Args:
        X_train (list of lists): Training feature matrix.
        
    Returns:
        scale_params (list of tuples): List of (min, max) for each feature.
    """
    num_features = len(X_train[0])
    scale_params = []
    
    for j in range(num_features):
        col_vals = [row[j] for row in X_train]
        min_val = min(col_vals)
        max_val = max(col_vals)
        scale_params.append((min_val, max_val))
        
    return scale_params

def transform_min_max(X, scale_params):
    """
    Scales features to the range [0, 1] using min and max from fit_min_max.
    
    Args:
        X (list of lists): Feature matrix.
        scale_params (list of tuples): (min, max) parameters for each feature.
        
    Returns:
        X_scaled (list of lists): Scaled feature matrix.
    """
    X_scaled = []
    for row in X:
        scaled_row = []
        for j, val in enumerate(row):
            min_val, max_val = scale_params[j]
            # Handle division by zero if all values are identical
            if max_val == min_val:
                scaled_val = 0.0
            else:
                scaled_val = (val - min_val) / (max_val - min_val)
            scaled_row.append(scaled_val)
        X_scaled.append(scaled_row)
    return X_scaled

def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Splits features and targets into random train and test subsets.
    
    Args:
        X (list of lists): Feature matrix.
        y (list): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.
        seed (int): Random seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    n = len(X)
    indices = list(range(n))
    
    # Use standard library Random class with a seed for deterministic shuffling
    rng = random.Random(seed)
    rng.shuffle(indices)
    
    split_idx = int(n * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test
