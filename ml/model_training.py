from typing import List, Tuple, Dict

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ml.data_preparation import split_dataset, prepare_features
from ml.model_validation import calculate_classification_metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
import numpy as np


def evaluate_models(data_labels: List[Tuple[dict, int]]) -> Dict[str, Dict[str, float]]:
    train_data, train_labels, test_data, test_labels = split_dataset(data_labels, train_size=0.7)

    # Convert dictionary features to numpy arrays
    train_data = prepare_features(train_data)
    test_data = prepare_features(test_data)

    # Initialize models
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'MLP Classifier': MLPClassifier(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Classifier': SVC(probability=True),
        'AdaBoost': AdaBoostClassifier(),
        'Gaussian Naive Bayes': GaussianNB(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis()
    }

    results = {}

    for name, model in models.items():
        # Train the model
        model.fit(train_data, train_labels)

        # Predict labels and probabilities
        y_pred = model.predict(test_data)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(test_data)[:, 1]  # Probability estimates for the positive class
        else:
            y_scores = model.decision_function(test_data) if 'decision_function' in dir(model) else y_pred

        # Calculate metrics
        metrics = calculate_classification_metrics(test_labels, y_pred, y_scores)
        results[name] = metrics

    return results


def k_fold_cross_validation(model: BaseEstimator, data_labels: List[Tuple[Dict, int]], k: int = 10) -> Dict[str, float]:
    """
    Performs k-fold cross-validation on the given model and data, returning average metrics.

    Parameters:
        model (BaseEstimator): The sklearn model to be evaluated.
        data_labels (List[Tuple[Dict, int]]): Data as a list of (feature_dict, label) tuples.
        k (int): Number of folds for cross-validation.

    Returns:
        Dict[str, float]: A dictionary of the average metrics (accuracy, precision, recall, f1_score, roc_auc) across the k folds.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    metrics_accumulator = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': [],
        'TP': [],
        'TN': [],
        'FP': [],
        'FN': []
    }

    vectorizer = DictVectorizer(sparse=False)
    features = vectorizer.fit_transform([features for features, _ in data_labels])
    labels = np.array([label for _, label in data_labels])

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
        else:
            y_scores = model.decision_function(X_test) if 'decision_function' in dir(model) else y_pred


        fold_metrics = calculate_classification_metrics(y_test, y_pred, y_scores)
        for key in metrics_accumulator:
            metrics_accumulator[key].append(fold_metrics[key])

    # Calculate the average of the metrics across all folds
    average_metrics = {metric: np.mean(values) for metric, values in metrics_accumulator.items()}
    return average_metrics