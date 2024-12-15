import argparse
import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, normalize
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform

CLASSIFIER = "logistic_regression"
classifiers = {
        #"knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9 , 10, 50, 100]}),
        #"naive_bayes": (GaussianNB(), {}),  # No hyperparameters to tune
        #"decision_tree": (DecisionTreeClassifier(), {"max_depth": [1, 3, 5, 10, 50, 150]}),
        #"svm": (SVC(), {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 5, 10, 100]}),
        #"logistic_regression": (LogisticRegression(max_iter=5000), {"C": [0.1, 1, 10, 100]}),
        "random_forest": (RandomForestClassifier(random_state=42), {"n_estimators": [1, 10, 50, 100, 200, 500, 1000, 2000, 5000], "max_depth": [1, 5, 10, 20, 50, 75]}),
        #"xgboost": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {"n_estimators": randint(10, 200), "max_depth": randint(3, 10), "learning_rate": uniform(0.01, 0.3)}),
        #"lightgbm": (LGBMClassifier(), {"n_estimators": randint(10, 200), "max_depth": randint(3, 10), "learning_rate": uniform(0.01, 0.3)}),
        #"catboost": (CatBoostClassifier(verbose=0), {"iterations": randint(10, 200), "depth": randint(3, 10), "learning_rate": uniform(0.01, 0.3)}),
        #"neural_network": (MLPClassifier(max_iter=10000, random_state=42), {"hidden_layer_sizes": [(50,), (100,), (50, 50), (50, 50)], "activation": ["relu", "tanh"], "alpha": [0.00005, 0.0001, 0.001, 0.01]})
    }

# Define classifiers
def get_classifier_and_params(name,classifiers):    
    return classifiers.get(name)

# Automatic Feature Selection
def select_features(data, labels, k=4):
    selector = SelectKBest(score_func=f_classif, k=k)
    data_selected = selector.fit_transform(data, labels)
    return data_selected, selector

def soft_acc(labels):
    labels[labels=='mostly-true'] = 'true'
    labels[labels=='barely-true'] = 'undefined'
    labels[labels=='half-true'] = 'undefined'
    labels[labels=='pants-fire'] = 'false'
    return labels

def construct_features(df, numb_questions=10):
    rows = df.shape[0]
    numb_claims = int(rows/numb_questions)
    processed_df = {}
    all_processed_rows = []
    for i in range(0, numb_claims):
        index = i*numb_questions
        processed_df['claim'] = df.iloc[index]['claim']
        processed_df['label'] = df.iloc[index]['label']
        processed_df['FHYes'] = df.iloc[index:index+(int(numb_questions/2))]['justification_explanation_verdict'].value_counts().get('Yes',0)
        processed_df['FHNo'] = df.iloc[index:index+(int(numb_questions/2))]['justification_explanation_verdict'].value_counts().get('No',0)
        processed_df['FHUnverified'] = df.iloc[index:index+(int(numb_questions/2))]['justification_explanation_verdict'].value_counts().get('Unverified',0)
        processed_df['SHYes'] = df.iloc[index+(int(numb_questions/2)):(index+10)]['justification_explanation_verdict'].value_counts().get('Yes',0)
        processed_df['SHNo'] = df.iloc[index+(int(numb_questions/2)):(index+10)]['justification_explanation_verdict'].value_counts().get('No',0)
        processed_df['SHUnverified'] = df.iloc[index+(int(numb_questions/2)):(index+10)]['justification_explanation_verdict'].value_counts().get('Unverified',0)
        all_processed_rows.append(processed_df.copy())
    df1 = pd.DataFrame(all_processed_rows)
    return df1

def feature_engineering(df):
    data = df
    #Polinomial transformation to 
    #poly = PolynomialFeatures(degree=2, include_bias=False)
    #data = poly.fit_transform(df)

    #Dimensionality reduction
    #pca = PCA(n_components=2)
    #pca_features = pca.fit_transform(df)
    #data = np.hstack([df, pca_features])

    #noise = np.random.normal(0, 0.01, df.shape)
    #data_with_noise = df + noise
    #data = np.hstack([df, data_with_noise])
    

    return data

#Balance dataset to get equal number of instances per class
def balance_dataset(df, numb_questions = 10):
    classes = df['label'].value_counts()
    labels = list(classes.index)
    instances = [classes.min()]*len(labels)
    balanced_df = pd.DataFrame(columns=df.columns)
    i = 0
    while sum(instances) > 0:        
        claim_questions = df.iloc[i*numb_questions:(i*numb_questions)+numb_questions]
        label_index = labels.index(claim_questions.iloc[0]['label'])
        if instances[label_index] > 0:
            balanced_df = balanced_df.append(claim_questions)
            instances[label_index] = instances[label_index] - 1
        i = i + 1
    return balanced_df

def main(args):
    data_dir = os.path.dirname(args.input_path)    
    np.random.seed(42)
    if(len(args.input_path.split('.csv'))>0):
        original_data = pd.read_csv(args.input_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip')
    else:
        original_data = pd.read_json(args.input_path, lines=True)
        
    
    # Preprocessing
    #original_data = balance_dataset(original_data)
    original_data = construct_features(original_data)
    data = original_data.drop(['claim', 'label'], axis=1)
    labels = original_data['label']
    data = feature_engineering(data)
    data_scaled = data

    #SMOTE (Synthetic Minority Oversampling Technique) to generate additional synthetic samples for underrepresented classes in dataset. 
    #This will help balance the dataset and potentially improve classifier accuracy
    smote = SMOTE(random_state=42)
    data_scaled, labels = smote.fit_resample(data_scaled, labels)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_scaled)
    data_scaled = normalize(data_scaled)
    
    

    labels = soft_acc(labels)
    #labels = original_data['label']

    all_classifiers = list(classifiers.keys())
    max_cv_scores = 0
    for trial_classifier in all_classifiers:
        # Initialize the classifier
        classifier_and_params = get_classifier_and_params(trial_classifier, classifiers)
        if classifier_and_params is None:
            raise ValueError(f"Classifier '{trial_classifier}' is not recognized.")    
        classifier, param_grid = classifier_and_params

        # Hyperparameter optimization
        print(f"'Optimizing hyperparameters for : {trial_classifier}")
        if param_grid:
            grid_search = GridSearchCV(classifier, param_grid, cv=6, scoring='accuracy')
            #grid_search = RandomizedSearchCV(classifier, param_grid, n_iter=50, cv=5, scoring='accuracy', random_state=42)
            grid_search.fit(data_scaled, labels)
            best_classifier = grid_search.best_estimator_
        else:
            best_classifier = classifier
        cv_scores = cross_val_score(best_classifier, data_scaled, labels, cv=5, scoring='accuracy')
        metric = cv_scores.max()
        if metric > max_cv_scores:
            chosen_classifier = {'classifier_model': trial_classifier, 'classifier': grid_search, 'scores':cv_scores}
            max_cv_scores = metric
        print(f"Cross-Validation Accuracy Scores: {cv_scores}")
        print(f"Accuracy: {cv_scores.max():.2f}")


    # Print results
    print(f"Classifier: {chosen_classifier['classifier_model']}")
    print(f"Classifier: {chosen_classifier['classifier'].best_params_}")
    print(f"Cross-Validation Accuracy Scores: {chosen_classifier['scores']}")
    print(f"Accuracy: {chosen_classifier['scores'].max():.2f}")

    return chosen_classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    main(args)