import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

CLASSIFIER = "random_forest"

# Define classifiers
def get_classifier_and_params(name):
    classifiers = {
        "knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
        "naive_bayes": (GaussianNB(), {}),  # No hyperparameters to tune
        "decision_tree": (DecisionTreeClassifier(), {"max_depth": [3, 5, 10]}),
        "svm": (SVC(), {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}),
        "logistic_regression": (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
        "random_forest": (RandomForestClassifier(random_state=42), {"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10]})
    }
    return classifiers.get(name)

# Automatic Feature Selection
def select_features(data, labels, k=4):
    selector = SelectKBest(score_func=f_classif, k=k)
    data_selected = selector.fit_transform(data, labels)
    return data_selected, selector

def soft_acc(labels):
    labels[labels=='mostly-true'] = 'true'
    labels[labels=='barely-true'] = 'neutral'
    labels[labels=='half-true'] = 'neutral'
    labels[labels=='pants-fire'] = 'false'
    return labels

def main(args):
    if(len(args.input_path.split('.csv'))>0):
        original_data = pd.read_csv(args.input_path,delimiter=',', encoding="utf_8")
    else:
        original_data = pd.read_json(args.input_path, lines=True)
    
    # Preprocessing
    data = original_data.drop(['Claim', 'Label'], axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    labels = soft_acc(original_data['Label'])

    # Initialize the classifier
    classifier_and_params = get_classifier_and_params(CLASSIFIER)
    if classifier_and_params is None:
        raise ValueError(f"Classifier '{CLASSIFIER}' is not recognized.")    
    classifier, param_grid = classifier_and_params

    # Hyperparameter optimization
    if param_grid:
        grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(data_scaled, labels)
        best_classifier = grid_search.best_estimator_
        print(f"Best Parameters for {CLASSIFIER}: {grid_search.best_params_}")
    else:
        best_classifier = classifier
    cv_scores = cross_val_score(best_classifier, data_scaled, labels, cv=5, scoring='accuracy')

    # Print results
    print(f"Classifier: {CLASSIFIER}")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)