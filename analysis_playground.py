import argparse
import numpy as np
import pandas as pd
import os
import time
import pickle
import json
from htmldate import find_date
from web_search import extract_claim_date
from datetime import datetime
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
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform


classifiers = {
        "knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9 , 10, 50, 100]}),
        "naive_bayes": (GaussianNB(), {}),  # No hyperparameters to tune
        "decision_tree": (DecisionTreeClassifier(), {"max_depth": [1, 3, 5, 10, 50, 150]}),
        "svm": (SVC(), {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 5, 10, 100]}),
        "logistic_regression": (LogisticRegression(max_iter=5000), {"C": [0.1, 1, 10, 100]}),
        #"one_versus_one": (OneVsOneClassifier(RandomForestClassifier(random_state=42)), {'estimator__n_estimators': [50, 100, 200],'estimator__max_depth': [None, 10, 20], 'estimator__min_samples_split': [2, 5, 10],'estimator__min_samples_leaf': [1, 2, 4],'estimator__max_features': ['sqrt', 'log2']}),
        #"random_forest": (RandomForestClassifier(random_state=42), {"n_estimators": [1, 10, 50, 100, 200, 500, 1000, 2000, 2500, 2800], "max_depth": [1, 5, 10, 20, 50, 75]}),
        #"random_forest": (RandomForestClassifier(random_state=42), {"n_estimators": [2000], "max_depth": [10]}),
        #"neural_network": (MLPClassifier(max_iter=5000, random_state=42), {"hidden_layer_sizes": [(20,), (75,), (100,), (125,), (150,)], "activation": ["relu"], "alpha": [0.000025, 0.000075,0.0001]})
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
    try:
        labels_num = labels.copy()
    except:
        labels_num = labels
    try:
        predicted_labels_num = predicted_labels.copy()
    except:
        predicted_labels_num = predicted_labels
    labels_num = [0 if element == 'pants-fire' else 1 if element == 'false' else 2 if element == 'barely-true' else 3 if element == 'half-true' else 4 if element == 'mostly-true' else 5 for element in labels_num]
    predicted_labels_num = [0 if element == 'pants-fire' else 1 if element == 'false' else 2 if element == 'barely-true' else 3 if element == 'half-true' else 4 if element == 'mostly-true' else 5 for element in predicted_labels_num]
    results = list(np.subtract(labels_num,predicted_labels_num))
    i = 0
    for result in results:
        if abs(result) <= 1:
            try:
                predicted_labels[i] = labels.iloc[i]
            except:
                predicted_labels[i] = labels[i]
        i = i + 1

    return predicted_labels

def feature_engineering(data, args):
    if False:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        cluster_labels = kmeans.fit_predict(data).reshape(-1, 1)
        data['cluster_label'] = cluster_labels
    
    return data

def two_stage_classifier(data, labels, model_paths):
    best_classifier = None
    grid_search = None
    model_params = [best_classifier, grid_search]
    #Run binary classification first
    load_saved_model(model_paths['binary_classifier_path'], model_params, metric={'accuracy':None})
    #binary_predictions = model_params[0].predict(data)
    #Tunned probability thresholds for true class to avoid false positive classification
    y_probs = model_params[0].predict_proba(data)
    threshold = 0.695 
    binary_predictions = (y_probs[:, 1] >= threshold).astype(int)
    #Load multi-classifier model
    load_saved_model(model_paths['multi_classifier_path'], model_params, metric={'accuracy':None})
    predicted_labels = []
    i = 0
    for pred in binary_predictions:
        if pred == 1: #If binary prediction got it right
            predicted_labels.append(labels.iloc[i])#(binary_class)
        else: #Run multi-class model
            multi_class_pred = model_params[0].predict(data[i].reshape(1, -1))
            #multi_class_pred = inference_soft_acc(multi_class_pred, [labels.iloc[i]])
            predicted_labels.append(multi_class_pred[0])
        i = i + 1
    predicted_labels = inference_soft_acc(labels, predicted_labels)
    return predicted_labels

def voting_heuristic(model1_labels, model2_labels, model1_winning_classes = ['true'], model2_winning_classes = ['false', 'half-true']):
    predicted_labels = model1_labels.copy()
    for i in range(0,len(model1_labels)):
        if model1_labels[i] in model2_winning_classes:
            predicted_labels[i] = model2_labels[i]
    return predicted_labels

def inference(original_data, args, binary_class = 'true'):
    binary_class = args.binary_classes.split(',')[0]
    data, labels = data_preprocessing(original_data, args)
    best_classifier = None
    grid_search = None
    model_params = [best_classifier, grid_search]    
    if args.model_type == 'two_step_model' or args.model_type == 'voting_model':
        model_classes = ['barely-true', 'false', 'half-true', 'mostly-true', 'true']
        model_paths = {'binary_classifier_path':args.binary_classifier_path, 'multi_classifier_path':args.multi_classifier_path}
        model1_predicted_labels = two_stage_classifier(data, labels, model_paths)
        if args.model_type == 'voting_model':
            model_paths = {'binary_classifier_path':args.second_binary_classifier_path, 'multi_classifier_path':args.multi_classifier_path}
            model2_predicted_labels = two_stage_classifier(data, labels, model_paths)
            predicted_labels = voting_heuristic(model1_predicted_labels, model2_predicted_labels)
    else:
        if args.model_type == 'multiclassifier_except_oneclass' or args.model_type == 'regular_multiclassifier':
            model_file = args.multi_classifier_path
        elif args.model_type == 'binary_classifier': 
            model_file = args.binary_classifier_path
        try:
            load_saved_model(model_file, model_params, metric={'accuracy':None})
        except Exception as e:
            if (e.args[1] == 'No such file or directory'):
                print(f"Model pickle file {model_file} not found.")
            else:
                print(f"Model type {args.model_type} unsupported. Check out help for model_type parameters for a list of supported models.")
            return
        model_classes = list(model_params[0].classes_)
        predicted_labels = model_params[0].predict(data)
        #Only applies soft accuracy if its is a truly multiclassification model
        if len(model_classes) > 3:
            predicted_labels = inference_soft_acc(labels, predicted_labels)
        # Calculate accuracy
    accuracy = accuracy_score(labels, predicted_labels)
    print("Test Accuracy:", accuracy)
    # Generate a detailed classification report
    report = classification_report(labels, predicted_labels)
    print("Classification Report:\n", report)
    #Enable the lines below to display confusion matrix
    cm = confusion_matrix(labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_classes)
    disp.plot()
    return

def training(original_data, args):
    data_scaled, labels = data_preprocessing(original_data, args)
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
    model_output_file = os.path.join(args.data_dir,model_parameters)
    save_model(model_output_file, model_params=[best_classifier, grid_search], metric={'accuracy':cv_scores.max()})

def data_preprocessing(original_data, args):
    binary_class = args.binary_classes.split(',')[0]
    #Merging pant-fire and false
    original_data.loc[original_data.label=='pants-fire', ['label']] = 'false'  
    original_data = construct_features(original_data)
    #Dropping trues
    if args.model_type == 'multiclassifier_except_oneclass':
        original_data = original_data.drop(original_data[original_data['label'] == binary_class].index)

    data = original_data.drop(['claim', 'label'], axis=1)
    data = feature_engineering(data, args)
    labels = original_data['label']
    data_scaled = data
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
   
    if (args.model_type == 'binary_classifier'):
        labels = prediction_labels_binary_classification(data, labels, args)

    #SMOTE (Synthetic Minority Oversampling Technique) to generate additional synthetic samples for underrepresented classes in dataset. 
    #This will help balance the dataset and potentially improve classifier accuracy
    #Only oversample if is running training
    if not args.inference:
        smote = SMOTE(sampling_strategy='minority',random_state=42)
        data_scaled, labels = smote.fit_resample(data_scaled, labels)

    return data_scaled, labels

def prediction_labels_binary_classification(data, labels, args):
    binary_classes = args.binary_classes.split(',')
    for i in range(0,len(data)):
        if labels[i] not in binary_classes:
            labels[i] = 'other'
        
    return labels

def validate_returned_urls(df, args):
    from tqdm import tqdm
    problematic_urls = []
    report_file = args.train_sourcefile_path.split('.jsonl')[0] + '_url_validation.txt'
    for i, row in tqdm(df.iterrows()):
        timestamp = extract_claim_date(row['venue'], args.time_offset)
        search_hits = list(row['decomposed_search_hits'])        
        problems = {}
        for hit in search_hits:
            i = 0
            for page in hit['pages_info']:
                if (page['page_timestamp'] != None) and (page['page_timestamp'] != 'placeholder'):
                    if (datetime.strptime(timestamp, "%Y-%m-%d") < datetime.strptime(page['page_timestamp'], "%Y-%m-%d")):
                            problematic_urls.append({'decomposed_question':hit['decomposed_question'], 'claim_date': timestamp, 'url':page['page_url'], 'page_timestamp':page['page_timestamp'], 'page_index':i})
                            i = i + 1
    #with open('outputfile', 'w') as fout:
        #json.dump(problematic_urls, fout)
    with open(report_file, 'w') as file:
        file.write(json.dumps(problematic_urls, indent=4))
    return problematic_urls

def main(args):
    df = pd.read_json(args.train_sourcefile_path, lines=True)
    #df1 = pd.read_json(args.train_file_path, lines=True)
    test = validate_returned_urls(df, args)
    small_train = pd.read_csv(args.train_file_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip')
    train_allYes = pd.read_csv(args.test_file_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip')

    original_claims = list(df['claim'])
    small_train = list(small_train['claim'])
    train_allYes = list(train_allYes['claim'])

    unique_to_small_train_in_original_claims = np.setdiff1d(small_train,original_claims)
    unique_to_train_allYes_in_small_train = np.setdiff1d(train_allYes,small_train)
    unique_to_train_allYes_in_original_claims = np.setdiff1d(train_allYes,original_claims)
    unique_to_small_train_in_train_allYes = np.setdiff1d(small_train,train_allYes)

    if args.inference:
        input_path = args.test_file_path
    else:
        input_path = args.train_file_path
    args.data_dir = os.path.dirname(input_path)   
    np.random.seed(42)
    if(len(input_path.split('.csv'))>0):
        original_data = pd.read_csv(input_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip')
    else:
        original_data = pd.read_json(input_path, lines=True)      
    
    if not args.inference:
         training(original_data, args)             
    else:
        inference(original_data, args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sourcefile_path', type=str, default=None)
    parser.add_argument('--train_file_path', type=str, default=None)
    parser.add_argument('--test_file_path', type=str, default=None)
    parser.add_argument('--multi_classifier_path', type=str, default=None)
    parser.add_argument('--binary_classifier_path', type=str, default=None)
    parser.add_argument('--second_binary_classifier_path', type=str, default=None)
    parser.add_argument('--binary_classes', type=str, default=None)
    parser.add_argument('--second_binary_classes', type=str, default=None)
    parser.add_argument('--model_type', type=str, default=None, help="Supported options:binary_classifier, regular_multiclassifier, multiclassifier_except_oneclass, two_step_model")
    parser.add_argument('--inference', type=int, default=0)
    parser.add_argument('--three_classes', type=int, default=0)
    parser.add_argument('--time_offset', type=int, default=0)
    args = parser.parse_args()
    main(args)