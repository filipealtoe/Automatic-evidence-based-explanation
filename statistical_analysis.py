import argparse
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import ttest_1samp, f_oneway, kruskal, chi2_contingency

def descriptive_stats(data, column_to_analyze):
    print(data[column_to_analyze].describe())
    # Histogram
    plt.hist(data[column_to_analyze], bins=20, alpha=0.7)
    title_ = column_to_analyze + 'Distribution'
    plt.title(title_)
    plt.xlabel(column_to_analyze)
    plt.ylabel('Frequency')
    plt.show()

def one_sample_t_test(data, threshold, column_to_analyze):
    t_stat, p_value = ttest_1samp(data[column_to_analyze], threshold)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    return t_stat, p_value

def outlier_detection(data, column_to_analyze, low_quantile=0.25, high_quantile=0.75):
    Q1 = data[column_to_analyze].quantile(low_quantile)
    Q3 = data[column_to_analyze].quantile(high_quantile)
    IQR = Q3 - Q1
    outliers = data[(data[column_to_analyze] < Q1 - 1.5 * IQR) | (data[column_to_analyze] > Q3 + 1.5 * IQR)]
    print(f"Outliers:\n{outliers}")
    return outliers

def anova_test(data, column_to_analyze):
    groups = [data[data[args.label_column] == label][column_to_analyze] for label in data[args.label_column].unique()]
    f_stat, p_value = f_oneway(*groups)
    print(f"F-statistic: {f_stat}, P-value: {p_value}")
    return f_stat, p_value

def krustal_test(data, column_to_analyze):
    groups = [data[data[args.label_column] == label][column_to_analyze] for label in data[args.label_column].unique()]
    f_stat, p_value = kruskal(*groups)
    print(f"Kruskal-Wallis H-statistic: {f_stat}, P-value: {p_value}")
    return f_stat, p_value

def chi_square_correlation(data, column_to_analyze):
    data['similarity_category'] = pd.cut(data[column_to_analyze], bins=3, labels=['low', 'medium', 'high'])
    contingency_table = pd.crosstab(data[args.label_column], data['similarity_category'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Square: {chi2}, P-value: {p_value}")
    return chi2, p_value

def visualize(x_axis, y_axis, data):
    box_title = y_axis + ' by ' + x_axis
    sns.boxplot(x=x_axis, y=y_axis, data=data)
    plt.title(box_title)
    plt.show()
    violin_title = y_axis + ' Distribution by ' + x_axis
    sns.violinplot(x=x_axis, y=y_axis, data=data)
    plt.title(violin_title)
    plt.show()

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

def summary_rouge_1(data):
    data = data.dropna(axis=0, subset=['generated_article_summary'])
    references = data['human_summary']
    generated_summaries = data['generated_article_summary']
    if len(references) != len(generated_summaries):
        raise ValueError("The number of references and generated summaries must be the same.")
    
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    n = len(references)
    
    for ref, gen in zip(references, generated_summaries):
        # Tokenize and count unigrams
        reference_tokens = ref.split()
        generated_tokens = gen.split()
        
        reference_counts = Counter(reference_tokens)
        generated_counts = Counter(generated_tokens)
        
        overlap = sum((reference_counts & generated_counts).values())
        
        # Calculate individual recall, precision, and F1-score
        recall = overlap / len(reference_tokens) if reference_tokens else 0.0
        precision = overlap / len(generated_tokens) if generated_tokens else 0.0
        f1_score = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
        
        total_recall += recall
        total_precision += precision
        total_f1 += f1_score
    
        # Compute average scores
        avg_recall = total_recall / n
        avg_precision = total_precision / n
        avg_f1_score = total_f1 / n
        
        return {
            "Average ROUGE-1 Recall": avg_recall,
            "Average ROUGE-1 Precision": avg_precision,
            "Average ROUGE-1 F1-Score": avg_f1_score
        }


def main(args):
    run_start_time = time.time()
    df_original = pd.read_json(args.input_path, lines=True)
    if args.five_classes:
        df_original.loc[df_original[args.label_column]=='pants-fire', [args.label_column]] = 'false'  
    if args.four_classes:
        df_original.loc[df_original[args.label_column]=='pants-fire', [args.label_column]] = 'false'
        df_original.loc[df_original[args.label_column]=='mostly-true', [args.label_column]] = 'true'
    if args.policy_subcategory:
        df_corpus = pd.read_json(args.corpus_file_path, lines=True)
        policy_claims = df_corpus.loc[df_corpus['subcategory']=='Politics']['claim']
        df_original = df_original.loc[df_original['claim'].isin(policy_claims)]
    columns_to_analyze = get_array_params(args.columns_to_analyze)
    #df_original = df_original[df_original['article_summary_similarity'].notna()]
    for column_to_analyze in columns_to_analyze:
        df = df_original.dropna(axis=0, subset=[column_to_analyze])
        #comparison_file_df = pd.read_json(args.comparison_file_path, lines=True)
        start = 0 if not args.start else args.start
        if not args.end:
            end = len(df)
        else:
            if args.end > len(df):
                end = len(df)
        df = df[start:end]
        
        if args.descriptive_stats:
            descriptive_stats(df, column_to_analyze)
        if args.one_sample_t_test:
            one_sample_t_test(df, args.similarity_threshold, column_to_analyze)
        if args.outlier_detection:
            outlier_detection(df, column_to_analyze)
        #Test if column_to_analyze differs significantly across labels
        if args.krustal_test:
            krustal_test(df, column_to_analyze)
        if args.chi_square_correlation:
            chi_square_correlation(df, column_to_analyze)
        #Box and Violin plots
        if args.visualization:
            visualize(args.label_column, column_to_analyze, df)
        rouge_scores = summary_rouge_1(df)
        print(rouge_scores)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--corpus_file_path', type=str, default=None)
    parser.add_argument('--columns_to_analyze', type=str, default='similarity_index')
    parser.add_argument('--label_column', type=str, default='human_label')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--similarity_threshold', type=float, default=None)
    parser.add_argument('--descriptive_stats', type=int, default=None)    
    parser.add_argument('--one_sample_t_test', type=int, default=None)
    parser.add_argument('--outlier_detection', type=int, default=None)
    parser.add_argument('--chi_square_correlation', type=int, default=None)
    parser.add_argument('--krustal_test', type=int, default=None)
    parser.add_argument('--visualization', type=int, default=None)
    parser.add_argument('--four_classes', type=int, default=0)
    parser.add_argument('--five_classes', type=int, default=1)
    parser.add_argument('--policy_subcategory', type=int, default=0)
    args = parser.parse_args()
    main(args)