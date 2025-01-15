import argparse
import numpy as np
import pandas as pd
import os
import time
import pickle
import json
import glob
from tqdm import tqdm
import uuid
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
import models.raw_evidence_retriever as retriever

MONTH_MAPPING = {"1": "January",
                 "01": "January",
                 "2": "February",
                 "02": "February",
                 "3": "March",
                 "03": "March",
                 "4": "April",
                 "04": "April",
                 "5": "May",
                 "05": "May",
                 "6": "June",
                 "06": "June",
                 "7": "July",
                 "07": "July",
                 "8": "August",
                 "08": "August",
                 "9": "September",
                 "09": "September",
                 "10": "October",
                 "11": "November",
                 "12": "December"
                 }

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

def merge_chunked_files(files_dir, consolidated_file_name, files_pattern='classification*.csv'):
    #df=pd.DataFrame()
    all_dfs = []
    jsonl_file = 0
    for file in glob.glob(f"{files_dir}/{files_pattern}"):
        #if file.endswith(files_extension):
        try:
            aux = pd.read_json(os.path.join(files_dir, file), lines=True)
            jsonl_file = 1
        except:
            aux=pd.read_csv(os.path.join(files_dir, file), encoding='utf-8', sep='\t', header=0)
        all_dfs.append(aux)
    df = pd.concat(all_dfs, axis=0)
    if not jsonl_file:
        df.to_csv(os.path.join(files_dir, consolidated_file_name), sep ='\t', header=True, index=False, encoding='utf-8')
    else:
        df.to_json(os.path.join(files_dir, consolidated_file_name), orient='records', lines=True)

def get_topic(dataset):
    for i, row in tqdm(dataset.iterrows()):
        scrapped_text = retriever.scrape_website_text(row['url'])

def format_claim_date(claim_date = ''):
        month_day_year = claim_date.split('/')
        try:
            month_day_year[1]
            date_str = "stated on {} {}, {}".format(MONTH_MAPPING[month_day_year[0]], month_day_year[1],  month_day_year[2])
        except:
            month_day_year = claim_date.split('-')
            date_str = "stated on {} {}, {}".format(MONTH_MAPPING[month_day_year[1]], month_day_year[2],  month_day_year[0])
        return date_str

def convert_dataset(datasettype, datasetpath):
    def format_finaldf(data_dict={}, data_values=[]):
        final_df = pd.DataFrame()
        for data_key,data_value in zip(data_dict.keys(),data_values):
            final_df[data_key] = data_value
        return final_df

    
    if datasettype == 'snopes':
        df = pd.read_csv(datasetpath,delimiter=',', encoding="utf_8", on_bad_lines='skip', dtype=str)
        df = df.dropna(subset=['article_date_phase1'])
        labels = df['fact_rating_phase1']
        claim_dates = df['article_date_phase1']
        final_labels = []
        example_ids = []
        venues = []
        persons = []
        for label,claim_date in zip(labels, claim_dates):
            example_ids.append(str(uuid.uuid4()))
            if label == 'mostly false':
                new_label = 'barely-true'
            elif label == 'FALSE':
                new_label = 'false'
            elif label == 'mixture':
                new_label = 'half-true'
            elif label == 'mostly true':
                new_label = 'mostly-true'
            else:
                new_label = 'true'
            venues.append(format_claim_date(claim_date))
            final_labels.append(new_label)
            persons.append('')
        
        '''
        final_df['example_id'] = example_ids
        final_df['label'] = final_labels
        final_df['url'] = df['snopes_url_phase1']
        final_df['claim'] = df['article_claim_phase1']
        final_df['category'] = df['article_category_phase1']
        final_df['person'] = persons
        final_df['venue'] = venues
        '''
        data_values = [example_ids, final_labels, df['snopes_url_phase1'], df['article_claim_phase1'], df['article_category_phase1'], persons, venues]
        data_dict = {'example_id':None, 'label':None, 'url':None, 'claim':None, 'category':None, 'person':None, 'venue':None}

    if datasettype == 'datacommons.org':
        df = pd.read_json(datasetpath, lines=True)
        df = df.dropna(subset=['datePublished'])
        labels = []
        claim_dates = df['datePublished']
        urls = df['url']
        claims = df['claimReviewed']
        example_ids = []
        venues = []
        persons = []
        categories = []
        subcategories = []
        fact_checkers = []
        labels = []
        for index,row in df.iterrows():
            example_ids.append(str(uuid.uuid4()))
            fact_checkers.append(row['author']['name'])
            label = row['reviewRating']['alternateName'].lower()
            if label == 'pants on fire':
                new_label = 'pants-fire'
            elif label == 'three pinocchios' or label == 'barely true':
                new_label = 'barely-true'
            elif label == 'four pinocchios':
                new_label = 'false'
            elif label == 'two pinocchios' or label == 'half true':
                new_label = 'half-true'
            elif label == 'one pinocchio' or label == 'mostly true' or label == 'not the whole story':
                new_label = 'mostly-true'
            elif label == 'geppetto checkmark':
                new_label = 'true'
            else: 
                new_label = label
            categories.append('Politics')
            subcategories.append('None')
            labels.append(new_label)
            venues.append(format_claim_date(claim_dates[index]))
            persons.append(row['itemReviewed']['author']['name'])
        
        data_values = [example_ids, labels, df['url'], df['claimReviewed'], categories, persons, venues, subcategories, fact_checkers]
        data_dict = {'example_id':None, 'label':None, 'url':None, 'claim':None, 'category':None, 'person':None, 'venue':None, 'subcategory':None, 'fact_checker':None}

    final_df = format_finaldf(data_dict, data_values)
    return final_df

def define_subcategory(original_dataset, new_dataset):
    original_claims = new_dataset['claim']
    subcategory = []
    for claim_text in original_claims:
        category = original_dataset[original_dataset['article_claim_phase1']==claim_text]['article_category_phase1']
        if str(category.iloc[0]).find('Politicians') != -1:
            subcategory.append('Politicians')
            continue
        try:
            category = str(category.iloc[0]).split('Politics  ')[1]
        except:
            category = str(category.iloc[0]).split('  Politics')[0]
        subcategory.append(category)
    new_dataset['subcategory'] = subcategory
    return new_dataset

def main(args):
    data_path = args.train_sourcefile_path
    merge_chunked_files(data_path,"poli_wp_snopes_mixed_test.csv", files_pattern="test*.csv")
    #original_train = pd.read_json(args.train_sourcefile_path, lines=True)
    #original_train = get_topic(original_train)
    #small_train = pd.read_json(args.train_file_path, lines=True)
    #df1 = pd.read_json(args.train_file_path, lines=True)
    #test = validate_returned_urls(df, args)
    #wp = pd.read_json(args.train_file_path, lines=True)
    #wp = pd.read_csv(args.train_file_path,delimiter=',', encoding="utf_8", on_bad_lines='skip', dtype=str)
    #snopes_politics = define_subcategory(snopes_checked_v03, snopes_politics)
    #snopes_politics.to_json(args.train_sourcefile_path, orient='records', lines=True)
    #wp = small_test.loc[small_test['author.name']=='Washington Post']
    small_test = pd.read_csv(args.test_file_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip', dtype=str)
    #small_test.shape
    final_df = convert_dataset('datacommons.org', args.train_sourcefile_path)
    final_df.to_json(args.output_path, orient='records', lines=True)
    #final_df = pd.read_json(args.train_file_path, lines=True)
    #politics = final_df[final_df['category'] == 'Politics']
    #politics.to_json(args.output_path, orient='records', lines=True)
    #wp = small_test.loc[small_test['author.name']=='Washington Post']
    #wp = wp.loc[wp['datePublished']!=None]
    #wp = wp.loc[wp['itemReviewed.@type']=='Claim']
    #fc = small_test.loc[small_test['author.name']=='FactCheck.org']
    #fc = fc.loc[fc['datePublished']!=None]
    #c = fc.loc[fc['itemReviewed.@type']=='Claim']
    #true_false = pd.read_csv(args.train_file_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip', dtype=str)
    #original_train = pd.read_csv(args.train_file_path,delimiter='\t', encoding="utf_8", on_bad_lines='skip', dtype=str)

    #small_train_1_477_claims = list(small_train_1_477['claim'])
    #small_train_claims = list(small_train['claim'])
    #test_103claims = list(test_103claims['claim'])

    #unique_to_test_103claims_in_small_test = np.setdiff1d(small_train_claims,small_train_1_477_claims)
    #unique_to_train_allYes_in_small_train = np.setdiff1d(train_allYes,small_train)
    #unique_to_train_allYes_in_original_claims = np.setdiff1d(train_allYes,original_claims)
    #unique_to_original_claims_in_small_claims = np.setdiff1d(original_claims,small_claims)

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sourcefile_path', type=str, default=None)
    parser.add_argument('--train_file_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
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