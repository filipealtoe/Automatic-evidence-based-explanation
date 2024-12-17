import argparse
import numpy as np
import pandas as pd
import os
import time
import pickle
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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform

CLASSIFIER = "logistic_regression"
classifiers = {
        "knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9 , 10, 50, 100]}),
        "naive_bayes": (GaussianNB(), {}),  # No hyperparameters to tune
        "decision_tree": (DecisionTreeClassifier(), {"max_depth": [1, 3, 5, 10, 50, 150]}),
        "svm": (SVC(), {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 5, 10, 100]}),
        "logistic_regression": (LogisticRegression(max_iter=5000), {"C": [0.1, 1, 10, 100]}),
        #"random_forest": (RandomForestClassifier(random_state=42), {"n_estimators": [1, 10, 50, 100, 200, 500, 1000, 2000, 2500, 2800], "max_depth": [1, 5, 10, 20, 50, 75]}),
        #"random_forest": (RandomForestClassifier(random_state=42), {"n_estimators": [2000], "max_depth": [10]}),
        #"neural_network": (MLPClassifier(max_iter=5000, random_state=42), {"hidden_layer_sizes": [(75,), (100,), (125,), (150,)], "activation": ["relu"], "alpha": [0.000025, 0.000075,0.0001]})
        #"xgboost": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), {"n_estimators": randint(10, 200), "max_depth": randint(3, 10), "learning_rate": uniform(0.01, 0.3)}),
        #"lightgbm": (LGBMClassifier(), {"n_estimators": randint(10, 200), "max_depth": randint(3, 10), "learning_rate": uniform(0.01, 0.3)}),
        #"catboost": (CatBoostClassifier(verbose=0), {"iterations": randint(10, 200), "depth": randint(3, 10), "learning_rate": uniform(0.01, 0.3)}),
        #"neural_network": (MLPClassifier(max_iter=5000, random_state=42), {"hidden_layer_sizes": [(10,), (20,), (50,), (100,), (10, 10)], "activation": ["relu", "tanh"], "alpha": [0.00005, 0.0001, 0.001, 0.01]})
    }

# Define classifiers
def get_classifier_and_params(name,classifiers):    
    return classifiers.get(name)

# Automatic Feature Selection
def select_features(data, labels, k=4):
    selector = SelectKBest(score_func=f_classif, k=k)
    data_selected = selector.fit_transform(data, labels)
    return data_selected, selector

def three_classes(labels):
    labels[labels=='mostly-true'] = 'true'
    labels[labels=='barely-true'] = 'undefined'
    labels[labels=='half-true'] = 'undefined'
    labels[labels=='pants-fire'] = 'false'
    return (labels, True)

def binary_classes(labels):
    labels[labels!='true'] = 'untrue'
    return (labels, True)

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

def save_model(pickle_file, model_params=[], metric={}):
    pickle_file = open(pickle_file,'wb')
    for model_param in model_params:
        pickle.dump(model_param,pickle_file)
    try:
        pickle.dump(metric['accuracy'],pickle_file)
        pickle.dump(metric['f1'],pickle_file)
    except Exception as e:
        pickle_file.close()
        return
    pickle_file.close()
    return

def load_saved_model(pickle_file, model_params=[], metric={}):
    pickle_file = open(pickle_file,'rb')
    for i in range(0,len(model_params)):
        model_params[i] = pickle.load(pickle_file)

    for metric in metric.keys():
        metric = pickle.load(pickle_file)
    pickle_file.close()

def inference_soft_acc(labels, predicted_labels):
    labels_num = labels.copy()
    predicted_labels_num = predicted_labels.copy()
    labels_num = [0 if element == 'pants-fire' else 1 if element == 'false' else 2 if element == 'barely-true' else 3 if element == 'half-true' else 4 if element == 'mostly-true' else 5 for element in labels_num]
    predicted_labels_num = [0 if element == 'pants-fire' else 1 if element == 'false' else 2 if element == 'barely-true' else 3 if element == 'half-true' else 4 if element == 'mostly-true' else 5 for element in predicted_labels_num]
    results = list(np.subtract(labels_num,predicted_labels_num))
    i = 0
    for result in results:
        if abs(result) <= 1:
            predicted_labels[i] = labels.iloc[i]
        i = i + 1

    return predicted_labels

def inference(args, labels, data, used_three_classes):
    best_classifier = None
    grid_search = None
    model_params = [best_classifier, grid_search]
    load_saved_model(args.multi_classifier_path, model_params, metric={'accuracy':None})
    predicted_labels = model_params[0].predict(data)
    display_labels = ['barely-true','false','half-true', 'mostly-true', 'true']
    #If original six classes were used calculate soft accuracy
    if not used_three_classes:
        predicted_labels = inference_soft_acc(labels, predicted_labels)
    # Calculate accuracy
    accuracy = accuracy_score(labels, predicted_labels)
    print("Test Accuracy:", accuracy)
    # Generate a detailed classification report
    report = classification_report(labels, predicted_labels)
    print("Classification Report:\n", report)
    cm = confusion_matrix(labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    #disp.plot()

def data_preprocessing(original_data, args):
    #testing merging pant-fire and false
    original_data.loc[original_data.label=='pants-fire', ['label']] = 'false'
    

    original_data = construct_features(original_data)
    #Dropping trues
    if not args.binary_classification:
        original_data = original_data.drop(original_data[original_data['label']=='true'].index)

    #Keeping only trues
    #original_data = original_data.drop(original_data[original_data['label']!='true'].index)
    
   

    data = original_data.drop(['claim', 'label'], axis=1)
    labels = original_data['label']
    data = feature_engineering(data)
    data_scaled = data
    #SMOTE (Synthetic Minority Oversampling Technique) to generate additional synthetic samples for underrepresented classes in dataset. 
    #This will help balance the dataset and potentially improve classifier accuracy
    #Only oversample if is running training
    if True:
        if not args.inference:
            smote = SMOTE(sampling_strategy='minority',random_state=42)
            #smote = SMOTE(sampling_strategy={'true': 90}, random_state=42)
            data_scaled, labels = smote.fit_resample(data_scaled, labels)

    data_scaled['label'] = labels
    #Shuffle expanded dataset
    if not args.inference:
        indices = np.random.permutation(len(data_scaled))
        data_scaled = data_scaled.iloc[indices]
    labels = data_scaled['label']
    data_scaled = data_scaled.drop(['label'], axis=1)    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_scaled)
    data_scaled = normalize(data_scaled) 
    
    #Three classes, False, True and Uncertain
    if args.three_classes:
        labels, used_three_classes = three_classes(labels)
    
    if args.binary_classification:
        labels[labels!='true'] = 'untrue'
    return data_scaled, labels

def main(args):
    used_three_classes = False
    data_dir = os.path.dirname(args.input_path)   
    np.random.seed(42)
    if(len(args.input_path.split('.csv'))>0):
        original_data = pd.read_csv(args.input_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip')
    else:
        original_data = pd.read_json(args.input_path, lines=True)   
    
    
    data_scaled, labels = data_preprocessing(original_data, args)

    if not args.inference:
        all_classifiers = list(classifiers.keys())
        stored_cv_scores = 0
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
                grid_search.fit(data_scaled, labels)
                best_classifier = grid_search.best_estimator_
            else:
                best_classifier = classifier
            cv_scores = cross_val_score(best_classifier, data_scaled, labels, cv=5, scoring='accuracy')
            metric = cv_scores.mean()
            if metric > stored_cv_scores:
                chosen_classifier = {'classifier_model': trial_classifier, 'classifier': grid_search, 'scores':cv_scores}
                stored_cv_scores = metric
            print(f"Cross-Validation Accuracy Scores: {cv_scores}")
            print(f"Accuracy: {stored_cv_scores:.2f}")

        # Print results
        print(f"Classifier: {chosen_classifier['classifier_model']}")
        print(f"Classifier: {chosen_classifier['classifier'].best_params_}")
        print(f"Cross-Validation Accuracy Scores: {chosen_classifier['scores']}")
        print(f"Max Accuracy: {chosen_classifier['scores'].max():.2f}")
        model_parameters = 'best_model_' + chosen_classifier['classifier_model'] + '_' + str(chosen_classifier['scores'].max()) + '_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl'
        model_output_file = os.path.join(data_dir,model_parameters)
        save_model(model_output_file, model_params=[best_classifier, grid_search], metric={'accuracy':cv_scores.max()})              
    else:
        inference(args, labels, data_scaled, used_three_classes)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--multi_classifier_path', type=str, default=None)
    parser.add_argument('--inference', type=int, default=0)
    parser.add_argument('--three_classes', type=int, default=0)
    parser.add_argument('--binary_classification', type=int, default=0)
    args = parser.parse_args()
    main(args)