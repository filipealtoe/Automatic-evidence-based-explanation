import argparse
import numpy as np
import pandas as pd
import os
import time
import pickle
import json
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
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt

CLASSIFIER = "logistic_regression"
classifiers = {
        "knn": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9 , 10, 50, 100]}),
        "naive_bayes": (GaussianNB(), {}),  # No hyperparameters to tune
        "decision_tree": (DecisionTreeClassifier(), {"max_depth": [1, 3, 5, 10, 50, 150]}),
        "svm": (SVC(), {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 5, 10, 100], "gamma":['scale', 'auto', 0.01, 0.1, 1]},),
        "logistic_regression": (LogisticRegression(max_iter=5000), {"C": [0.1, 1, 10, 100]}),
        #"one_versus_one": (OneVsOneClassifier(SVC(probability=True,random_state=42)), {"estimator__kernel": ["linear", "rbf", "poly"], "estimator__C": [0.1, 1, 5, 10, 100], "estimator__gamma":['scale', 'auto', 0.01, 0.1, 1]}),
        #"one_versus_rest": (OneVsRestClassifier(SVC(probability=True,random_state=42)), {"estimator__kernel": ["linear", "rbf", "poly"], "estimator__C": [0.1, 1, 5, 10, 100], "estimator__gamma":['scale', 'auto', 0.01, 0.1, 1]}),
        #"random_forest": (RandomForestClassifier(random_state=42), {"n_estimators": [1, 10, 50, 100, 200, 500, 1000, 2000, 2500, 2800], "max_depth": [1, 5, 10, 20, 50, 75]}),
        #"neural_network": (MLPClassifier(max_iter=5000, random_state=42), {"hidden_layer_sizes": [(20,), (75,), (100,), (125,), (150,)], "activation": ["relu"], "alpha": [0.000025, 0.000075,0.0001]})
        #"xgboost": (XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, objective='multi:softprob'), {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}),
        #"lightgbm": (LGBMClassifier(random_state=42), {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2], 'num_leaves': [15, 31, 63],'subsample': [0.8, 1.0],'min_child_samples': [10, 20, 50],'min_data_in_leaf': [10, 20, 50]}),
        #"catboost": (CatBoostClassifier(verbose=0, random_state=42), {"iterations": [50, 100, 200], "depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]}),
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

def inference_soft_acc(labels, predicted_labels, model_classes):
    '''
    if type(model_classes[0]) != str and type(model_classes[0]) != np.str_:
        labels_num = labels.copy()
        predicted_labels_num = predicted_labels.copy()
    else:
        labels_num = labels
        predicted_labels_num = predicted_labels
    '''
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
        else:
            try:
                predicted_labels[i].shape
                predicted_labels[i] = predicted_labels[i][0]
            except:
                pass
        i = i + 1

    return predicted_labels

def feature_engineering(data, args):
    if args.feature_engineering:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42)
        cluster_labels = kmeans.fit_predict(data).reshape(-1, 1)
        data['cluster_label'] = cluster_labels
    
    return data

def get_array_params(string_param = ''):
    string_param = string_param.replace(', ', ',')
    if string_param == None or string_param == 'Skip' or string_param == 'skip':
        return('skip')
    try:
        string_param.remove('')
    except:
        pass
    parameters = string_param.split(',')
    return parameters

def binary_classifier(data, model_path, emphasis_binary_class = 'true', pred_threshold=0.5):
    try:
        emphasis_binary_class.remove('')
    except:
        pass
    best_classifier = None
    grid_search = None
    model_params = [best_classifier, grid_search]
    #Run binary classification first
    load_saved_model(model_path, model_params, metric={'accuracy':None})
    binary_labels = model_params[0].classes_
    emphasis_binary_class_index = np.where(np.array(binary_labels) == emphasis_binary_class)[0][0]
    y_probs = model_params[0].predict_proba(data)
    binary_predictions = (y_probs[:, emphasis_binary_class_index] >= pred_threshold).astype(int)
    predicted_labels = []
    for pred in binary_predictions:
        if pred == 1: #If binary prediction got it right
            predicted_labels.append(binary_labels[emphasis_binary_class_index])
        else: 
            predicted_labels.append('other')
    return predicted_labels, binary_labels

def two_stage_classifier(original_data, labels, model_paths, args, emphasis_binary_class, pred_threshold=0.5):
    data, labels, mod_df = data_preprocessing(original_data, args)
    accuracy_calc = args.two_stage_acc_calc
    try:
        emphasis_binary_class.remove('')
    except:
        pass
    best_classifier = None
    grid_search = None
    model_params = [best_classifier, grid_search]
    #Run binary classification first
    load_saved_model(model_paths['binary_classifier_path'], model_params, metric={'accuracy':None})
    predicted_labels = []
    if emphasis_binary_class[0]!='skip':
        if len(emphasis_binary_class) == 1:
            binary_labels = model_params[0].classes_
            emphasis_binary_class_index = np.where(np.array(binary_labels) == emphasis_binary_class)[0][0]
            y_probs = model_params[0].predict_proba(data)
        #    threshold = 0.45
            binary_predictions = (y_probs[:, emphasis_binary_class_index] >= pred_threshold).astype(int)
            #Load multi-classifier model
            load_saved_model(model_paths['multi_classifier_path'], model_params, metric={'accuracy':None})
            #If model has numeric classes
            labels_encoded = 0
            model_classes = list(model_params[0].classes_) 
            if type(model_classes[0]) != str and type(model_classes[0]) != np.str_:
                original_labels = labels.copy()
                labels_encoded = 1
                multiclass_data, new_labels, encoded_labels, label_encoder_obj = encode_labels(original_data, args)
            else:
                multiclass_data = data
            i = 0
            for pred in binary_predictions:
                if pred == 1: #If binary prediction got it right
                    predicted_labels.append(binary_labels[emphasis_binary_class_index])#labels.iloc[i])#(binary_class)
                else: #Run multi-class model
                    multi_class_pred = model_params[0].predict(multiclass_data[i].reshape(1, -1))
                    if labels_encoded:
                        multi_class_pred = label_encoder_obj.inverse_transform(multi_class_pred)
                    predicted_labels.append(multi_class_pred[0])
                i = i + 1
        else:
            binary_predictions = model_params[0].predict(data)
            load_saved_model(model_paths['multi_classifier_path'], model_params, metric={'accuracy':None})
            #If model has numeric classes
            model_classes = list(model_params[0].classes_) 
            labels_encoded = 0
            if type(model_classes[0]) != str and type(model_classes[0]) != np.str_:
                original_labels = labels.copy()
                labels_encoded = 1
                multiclass_data, new_labels, encoded_labels, label_encoder_obj = encode_labels(original_data, args)
            else:
                multiclass_data = data
            i = 0
            for pred in binary_predictions:
                if pred in emphasis_binary_class: #If prediction is one of the emphasis classes, got it right
                    predicted_labels.append(pred)#labels.iloc[i])#(binary_class)
                else: #Run multi-class model
                    multi_class_pred = model_params[0].predict(multiclass_data[i].reshape(1, -1))
                    if labels_encoded:
                        multi_class_pred = label_encoder_obj.inverse_transform(multi_class_pred)
                    predicted_labels.append(multi_class_pred[0])
                i = i + 1
    #If skipping binary_classifier, refer back to a single stage model
    else:
        labels_encoded = 0
        load_saved_model(model_paths['multi_classifier_path'], model_params, metric={'accuracy':None})
        model_classes = list(model_params[0].classes_) 
        if type(model_classes[0]) != str and type(model_classes[0]) != np.str_:
            original_labels = labels.copy()
            labels_encoded = 1
            data, new_labels, encoded_labels, label_encoder_obj = encode_labels(original_data, args)
        predicted_labels = model_params[0].predict(data)
        if labels_encoded:
            predicted_labels = label_encoder_obj.inverse_transform(predicted_labels)
    #predicted_labels = inference_soft_acc(labels, predicted_labels)
    if accuracy_calc:
        predicted_labels_int = predicted_labels.copy()
        predicted_labels_int = inference_soft_acc(labels, predicted_labels_int, model_classes)
        report = classification_report(labels, predicted_labels_int)
        print("Classification Report:\n", report)
        #Enable the lines below to display confusion matrix
        cm = confusion_matrix(labels, predicted_labels_int)
        model_classes = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']
        if args.five_classes:
            model_classes = ['barely-true', 'false', 'half-true', 'mostly-true', 'true']
        if args.four_classes:
            model_classes = ['barely-true', 'false', 'half-true', 'true']
        #model_classes = model_params[0].classes_
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_classes)
        disp.plot()
    return predicted_labels

def voting_heuristic(model1_labels, model2_labels, args):
    model1_winning_classes = args.model_one_winning_classes.split(',')
    try:
        model1_winning_classes.remove('')
    except:
        pass
    predicted_labels = model1_labels.copy()
    for i in range(0,len(model1_labels)):
        if not model1_labels[i] in model1_winning_classes:
            try:
                predicted_labels[i] = model2_labels[i]
            except:
                predicted_labels[i] = model2_labels[i][0]
    return predicted_labels

def update_classification_report(test_dataset, args, labels, predicted_labels,
                                                  sub_categories_to_remove=['Imagery', 'Not Verifiable']):
    #test_file_path = os.path.join(os.path.dirname(args.test_file_path), 'datasets', os.path.basename(args.test_file_path).split('.')[0] + '.jsonl')
    test_file_path = args.corpus_file_path 
    original_dataset = pd.read_json(test_file_path, lines=True)
    claims = original_dataset.loc[original_dataset[args.stats_parameter].isin(sub_categories_to_remove)]['claim']
    data, labels, test_dataset = data_preprocessing(test_dataset, args)
    df = test_dataset.reset_index()
    indexes = df.loc[df['claim'].isin(claims)].index
    indexes = indexes//10
    #new_labels = labels.loc[set(labels.index) - set(indexes)]
    new_labels = labels.loc[[item for item in labels.index if item not in indexes]]
    new_predicted_labels = np.delete(predicted_labels, indexes, axis=0)       
        
    updated_report_str = classification_report(new_labels, new_predicted_labels) 
    removed_cats = ', '.join(sub_categories_to_remove)
    print('Removed Categories: {} - Classification Report:\n {}'.format(removed_cats, updated_report_str))  
        
    return updated_report_str

def plot_charts(results, output_file):

    all_data = []

    for cls, categories_data in results.items():
        # Create a subplot for the categories
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"Class {cls} - Distribution of Subcategories by Category", fontsize=16)
        
        # Flatten the axes for easier iteration
        axes = axes.flatten()
        
        # Iterate over the list of category dictionaries
        for i, category_data in enumerate(categories_data):
            # Safeguard for more categories than available axes
            if i >= len(axes):
                print(f"Skipping category due to lack of axes: {category_data}")
                continue
            
            # Process each category
           # for category, data in category_data.items():
            category = list(category_data.keys())[0]
            subcategories = category_data[category].values
            claims = category_data[list(category_data.keys())[1]].values
            
            # Count the occurrences of each subcategory
            subcategory_counts = {}
            subcategory_claims = {}
            for subcategory, claim in zip(subcategories, claims):
                subcategory_counts[subcategory] = subcategory_counts.get(subcategory, 0) + 1
                subcategory_claims.setdefault(subcategory, []).append(claim)
            
            # Calculate percentages
            total = sum(subcategory_counts.values())
            labels = list(subcategory_counts.keys())
            sizes = [(count / total) * 100 for count in subcategory_counts.values()]
            
            # Append data to the combined list
            for label, count, percentage in zip(labels, subcategory_counts.values(), sizes):
                all_data.append({
                    "Class": cls,
                    "Category": category,
                    "Subcategory": label,
                    "Count": count,
                    "Percentage": percentage,
                    "Claims": subcategory_claims[label]
                })
            
            # Plot the pie chart
            ax = axes[i]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title(category)
            ax.axis('equal')  # Equal aspect ratio for a circular pie chart
        
        # Adjust layout and show the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
        plt.show()
    
    # Save all data to a single JSON file
    df = pd.DataFrame(all_data, columns=list(all_data[0].keys()))
    df.to_json(output_file, orient='records', lines=True)
    return

def policy_only(dataset, args, subcategory='Politics'):
    original_dataset = pd.read_json(args.corpus_file_path, lines=True) 
    category_corpus_claims = original_dataset.loc[original_dataset['subcategory'] == subcategory]['claim']
    dataset = dataset.loc[dataset['claim'].isin(category_corpus_claims)]
    return dataset

def claim_analysis(test_dataset, args, classes, labels, predicted_labels):
    test_file_path = os.path.join(os.path.dirname(args.test_file_path), 'datasets', os.path.basename(args.test_file_path).split('.')[0] + '.jsonl')
    model_label = '_' + args.model_label + '_'
    stats_file_path = os.path.join(os.path.dirname(args.test_file_path), 'stats', os.path.basename(args.test_file_path).split('.')[0] + model_label + 'stats.jsonl')
    original_dataset = pd.read_json(test_file_path, lines=True)
    
    results = {}
    try:
        predicted_labels = predicted_labels[:,0]
    except:
        pass
    for cls in classes:
        # True Positives
        #tp_indices = ((labels.values == cls) & (predicted_labels== cls)[:,0])
        tp_indices = ((labels.values == cls) & (predicted_labels== cls))
        indices = np.where(tp_indices == True)
        tp_claims = test_dataset.iloc[indices]['claim']
        tp_categories = original_dataset.loc[original_dataset['claim'].isin(tp_claims)][args.stats_parameter]
        
        # False Positives
        #fp_indices = ((labels.values != cls) & (predicted_labels == cls)[:,0])
        fp_indices = ((labels.values != cls) & (predicted_labels == cls))
        indices = np.where(fp_indices == True)
        fp_claims = test_dataset.iloc[indices]['claim']
        fp_categories = original_dataset.loc[original_dataset['claim'].isin(fp_claims)][args.stats_parameter]
        
        # False Negatives
        #fn_indices = ((labels.values == cls) & (predicted_labels != cls)[:,0])
        fn_indices = ((labels.values == cls) & (predicted_labels != cls))
        indices = np.where(fn_indices == True)
        fn_claims = test_dataset.iloc[indices]['claim']
        fn_categories = original_dataset.loc[original_dataset['claim'].isin(fn_claims)][args.stats_parameter]

        
        # True Negatives
        #tn_indices = ((labels.values != cls) & (predicted_labels != cls)[:,0])
        tn_indices = ((labels.values != cls) & (predicted_labels != cls))
        indices = np.where(tn_indices == True)
        tn_claims = test_dataset.iloc[indices]['claim']
        tn_categories = original_dataset.loc[original_dataset['claim'].isin(tn_claims)][args.stats_parameter]

        results[cls] = [
            {'True Positives': tp_categories, 'claims': tp_claims},
            {'False Positives': fp_categories, 'claims': fp_claims},
            {'False Negatives': fn_categories, 'claims': fn_claims},
            {'True Negatives': tn_categories,'claims': tn_claims},
        ]
    
    #Retrieve missed prediction claims
    #missed_index = np.where(labels.values != predicted_labels[:,0])
    missed_index = np.where(labels.values != predicted_labels)
    missed_claims = test_dataset.iloc[missed_index]['claim']
    missed_categories = original_dataset.loc[original_dataset['claim'].isin(missed_claims)][args.stats_parameter]
    '''results['Missed Predictions'] = {
        'True Positives': ['None'],
        'False Positives': missed_categories,
        'False Negatives': ['None'],
        'True Negatives': ['None'],
    }'''
    
    if args.plot_charts:
        plot_charts(results, stats_file_path)
    return results, missed_claims

def inference(original_data, args, binary_class = 'true'):
    labels_encoded = 0
    binary_class = args.binary_classes.split(',')[0]
    data, labels, mod_df = data_preprocessing(original_data, args)
    best_classifier = None
    grid_search = None
    model_params = [best_classifier, grid_search]    
    if args.model_type == 'two_stage_classifier' or args.model_type == 'voting_model':
        model_classes = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']
        if args.four_classes:
            model_classes = ['barely-true', 'false', 'half-true', 'true']
        if args.five_classes:
            model_classes = ['barely-true', 'false', 'half-true', 'mostly-true', 'true']
        model_paths = {'binary_classifier_path':args.binary_classifier_path, 'multi_classifier_path':args.multi_classifier_path}
        model1_predicted_labels = two_stage_classifier(original_data, labels, model_paths, args, args.binary_classes.split(','), args.binary_prob_threshold)    
        predicted_labels = model1_predicted_labels 
        if args.model_type == 'voting_model':
            model_paths = {'binary_classifier_path':args.second_binary_classifier_path, 'multi_classifier_path':args.multi_classifier2_path}
            model2_predicted_labels = two_stage_classifier(original_data, labels, model_paths, args, args.second_binary_classes.split(','), args.second_binary_prob_threshold)
            predicted_labels = voting_heuristic(model1_predicted_labels, model2_predicted_labels, args)

    else:
        try:
            if args.model_type == 'multiclassifier_except_binaryclass' or args.model_type == 'regular_multiclassifier':
                model_file = args.multi_classifier_path
                load_saved_model(args.multi_classifier_path, model_params, metric={'accuracy':None})
                model_classes = list(model_params[0].classes_) 
                #If model has numeric classes
                if type(model_classes[0]) != str and type(model_classes[0]) != np.str_:
                    original_labels = labels.copy()
                    labels_encoded = 1
                    data, labels, encoded_labels, label_encoder_obj = encode_labels(original_data, args)
                predicted_labels = model_params[0].predict(data)
            elif args.model_type == 'binary_classifier': 
                model_file = args.binary_classifier_path
                #true binary
                if len(args.binary_classes)==2:
                    predicted_labels, model_classes = binary_classifier(data, args.binary_classifier_path, args.binary_classes, args.binary_prob_threshold)
                #multiclass
                else: 
                    load_saved_model(args.binary_classifier_path, model_params, metric={'accuracy':None})
                    model_classes = list(model_params[0].classes_)
                    predicted_labels = model_params[0].predict(data)
            #load_saved_model(model_file, model_params, metric={'accuracy':None})
        except Exception as e:
            if (e.args[1] == 'No such file or directory'):
                print(f"Model pickle file {model_file} not found.")
            else:
                print(f"Model type {args.model_type} unsupported. Check out help for model_type parameters for a list of supported models.")
            return
               
        #predicted_labels = model_params[0].predict(data)
        #Only applies soft accuracy if its is a truly multiclassification model
    if labels_encoded:
        labels = original_labels.copy()
        predicted_labels = label_encoder_obj.inverse_transform(predicted_labels)
    if len(model_classes) > 3:
        predicted_labels = inference_soft_acc(labels, predicted_labels, model_classes)
        # Calculate accuracy
    accuracy = accuracy_score(labels, predicted_labels)
    print("Test Accuracy:", accuracy)
    #Save classifed dataset
    save_classified_dataset(predicted_labels, mod_df, args)
    # Generate a detailed classification report
    report = classification_report(labels, predicted_labels)
    print("Classification Report:\n", report)
    #Enable the lines below to display confusion matrix
    cm = confusion_matrix(labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_classes)
    disp.plot()
    if args.generate_stats:
        category_contributions = claim_analysis(args.processed_dataset, args, model_classes, labels, predicted_labels)
    categories_to_remove = get_array_params(string_param = args.categories_to_remove)
    if categories_to_remove != 'skip':
        updated_report = update_classification_report(original_data, args, labels, predicted_labels,
                                                  sub_categories_to_remove=categories_to_remove)
    return

def encode_labels(dataset, args):
    numb_lab_original_data = dataset.copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(numb_lab_original_data['label'])    
    numb_lab_original_data['label'] = y_encoded
    data_scaled, labels, mod_df = data_preprocessing(numb_lab_original_data, args)
    return data_scaled, labels, y_encoded, label_encoder


def training(original_data, args):
    data_scaled, labels = data_preprocessing(original_data, args)
    all_classifiers = list(classifiers.keys())
    stored_cv_scores = 0
    for trial_classifier in all_classifiers:
        if trial_classifier == 'xgboost':
            data_scaled, labels, encoded_labels, label_encoder_obj = encode_labels(original_data, args)
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
        cv_scores = cross_val_score(best_classifier, data_scaled, labels, cv=6, scoring='accuracy')
        metric = cv_scores.mean()
        #metric = cv_scores.max()
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
    binary_class = args.binary_classes.split(',')
    #Merging pant-fire and false
    if args.five_classes:
        original_data.loc[original_data.label=='pants-fire', ['label']] = 'false'  
    if args.four_classes:
        original_data.loc[original_data.label=='pants-fire', ['label']] = 'false'
        original_data.loc[original_data.label=='mostly-true', ['label']] = 'true'
    original_data = construct_features(original_data)
    if args.policy_only:
        original_data = policy_only(original_data, args, subcategory='Politics')
    #remove duplicate claims if there are any
    #original_data = original_data.drop_duplicates(subset=['claim'])
    if args.generate_stats:
        args.processed_dataset = original_data
    #DeleteMe#####
    #test1 = [26,28,31,41,45,47,53,54]
    #original_data = original_data.drop(test1)
    #Dropping binary_class
    if args.model_type == 'multiclassifier_except_binaryclass':
        #original_data = original_data.drop(original_data[original_data['label'] == binary_class].index)
        original_data = original_data.drop(original_data[original_data.label.isin(binary_class)].index.tolist())

    data = original_data.drop(['claim', 'label'], axis=1)
    data = feature_engineering(data, args)
    labels = original_data['label']
    data_scaled = data
    data_scaled['label'] = labels
    #Shuffle training dataset
    if not args.inference:
        indices = np.random.permutation(len(data_scaled))
        data_scaled = data_scaled.iloc[indices]
    labels = data_scaled['label']
    data_scaled = data_scaled.drop(['label'], axis=1)    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_scaled)
    data_scaled = normalize(data_scaled) 
   
    if args.model_type == 'binary_classifier':
        labels = prediction_labels_binary_classification(data, labels, args)

    #SMOTE (Synthetic Minority Oversampling Technique) to generate additional synthetic samples for underrepresented classes in dataset. 
    #This will help balance the dataset and potentially improve classifier accuracy
    #Only oversample if is running training
    if not args.inference:
        #"minority", "not minority", "not majority", "all"
        smallest_class_size = min(original_data['label'].value_counts().values)
        #Protection for supper-small class sizes
        if smallest_class_size < 7:
            n_neighbors = int(smallest_class_size -1)
        else:
            n_neighbors = 6
        smote = SMOTE(sampling_strategy=args.SMOTE_type, k_neighbors=n_neighbors, random_state=42)
        data_scaled, labels = smote.fit_resample(data_scaled, labels)

    return data_scaled, labels, original_data

def prediction_labels_binary_classification(data, labels, args):
    binary_classes = args.binary_classes.split(',')
    for i in range(0,len(data)):
        if labels.iloc[i] not in binary_classes:
            labels.iloc[i] = 'other'
        
    return labels

def save_classified_dataset(predicted_labels, classified_dataset, args):
    file_path = os.path.join(args.output_file_dir, os.path.join(os.path.basename(args.test_file_path).split('.')[0] + '_classified.jsonl'))
    corpus_dataset = pd.read_json(args.corpus_file_path, lines=True)
    dataset_to_save = corpus_dataset.loc[corpus_dataset['claim'].isin(classified_dataset['claim'].values)]
    dataset_to_save['predicted_label'] = ''
    for i in range(0,classified_dataset.shape[0]):
        dataset_to_save.loc[dataset_to_save['claim'] == classified_dataset.iloc[i]['claim'],'predicted_label'] = predicted_labels[i][0]
    dataset_to_save.to_json(file_path, orient='records', lines=True)
    return

def main(args):
    used_three_classes = False
    if args.inference:
        input_path = args.test_file_path
    else:
        input_path = args.train_file_path
    args.data_dir = os.path.dirname(input_path)   
    np.random.seed(42)
    try:
        input_path.split('.csv')[1]
        original_data = pd.read_csv(input_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip', dtype=str)
    except Exception as e:
        original_data = pd.read_json(input_path, lines=True)
    '''
    if(len(input_path.split('.csv'))>0):
        original_data = pd.read_csv(input_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip')
    else:
        original_data = pd.read_json(input_path, lines=True)      
    '''
    if not args.inference:
         training(original_data, args)             
    else:
        inference(original_data, args)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_path', type=str, default=None)
    parser.add_argument('--test_file_path', type=str, default=None)
    parser.add_argument('--corpus_file_path', type=str, default=None)
    parser.add_argument('--output_file_dir', type=str, default=None)
    parser.add_argument('--multi_classifier_path', type=str, default=None)
    parser.add_argument('--multi_classifier2_path', type=str, default=None)
    parser.add_argument('--binary_classifier_path', type=str, default=None)
    parser.add_argument('--second_binary_classifier_path', type=str, default=None)
    parser.add_argument('--model_one_winning_classes', type=str, default=None)
    parser.add_argument('--binary_classes', type=str, default=None)
    parser.add_argument('--binary_prob_threshold', type=float, default=None)
    parser.add_argument('--second_binary_prob_threshold', type=float, default=None)
    parser.add_argument('--second_binary_classes', type=str, default=None)
    parser.add_argument('--model_type', type=str, default=None, help="Supported options:binary_classifier, regular_multiclassifier, multiclassifier_except_binaryclass, two_stage_classifier, voting_model")
    parser.add_argument('--inference', type=int, default=0)
    parser.add_argument('--two_stage_acc_calc', type=int, default=0)
    parser.add_argument('--feature_engineering', type=int, default=0)
    parser.add_argument('--claim_analysis', type=int, default=0)
    parser.add_argument('--five_classes', type=int, default=0)
    parser.add_argument('--four_classes', type=int, default=0)
    parser.add_argument('--three_classes', type=int, default=0)
    parser.add_argument('--generate_stats', type=int, default=0)
    parser.add_argument('--policy_only', type=int, default=0)
    parser.add_argument('--categories_to_remove', type=str, default=None)
    parser.add_argument('--plot_charts', type=int, default=0)
    parser.add_argument('--stats_parameter', type=str, default=None)
    parser.add_argument('--model_label', type=str, default=None)
    parser.add_argument('--SMOTE_type', type=str, default='all', help="Supported options: minority, not minority, not majority, all")
    args = parser.parse_args()
    main(args)