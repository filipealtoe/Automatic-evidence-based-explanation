# EFFORT: Explainable Framework for Fact-checking Oversight in Real-Time

## Getting started
Clone the repository and install the requirements:
```
pip install -r requirements.txt
```

Corpus datasets, autogenerated sample datasets, models pkls files, including their configuration parameters and confusion matrices are available via: https://drive.google.com/drive/folders/1KSni3hrkq8MCyQoOURnEEq7yfyY2qd9T?usp=sharing

## Corpus Data format
The corpus data files are formatted as jsonlines. The description of each field is as follows:  

| Field            | type         | Description                                                                                 |
|------------------|--------------|---------------------------------------------------------------------------------------------|
| `example_id`     | string       | Example ID                                                                                  |
| `claim`          | string       | Claim                                                                                       |
| `label`          | string       | Label: pants-fire, false, barely-true, half-true, mostly-true, true                         |
| `person`         | string       | Person who made the claim                                                                   |
| `venue`          | string       | Date and venue of the claim                                                                 |
| `url`            | string       | Politifact URL of the claim                                                                 |
| `category`       | string       | The claim category assigned by a LLM with in-context learning                               |
| `subcategory`    | string       | Specific to the Politics category, a subcategort assigned by a LLM with in-context learning |

## Autogenerated Sample Data format
Additionally to the corpus data format described above, the auto-generated dataset includes the following data fields: 

| Field                                   | type             | Description                                                                  |
|-----------------------------------------|------------------|------------------------------------------------------------------------------|
| `decomposed_questions`                  | List[string]     | The 10 decomposed question for the corresponding claim                       |
| `decomposed_justifications`             | List[string]     | The 10 decomposed justifications for the corresponding decomposed questions  |
| `decomposed_justification_explanations` | List[string]     | The 10 explanation summaries for the corresponding decomposed questions      |
| decomposed_search_hits                  | List[dictionary] | Includes a list with the evidence retrieved for each decomposed question     |

Each `decomposed_search_hits` element is a dictionary with the following fields:
```
decomposed_search_hit = {
    "decomposed_question: "The original decomposed question that initiated the evidence retrieval"
    "decomposed_justification: "The justification to reason the corresponding decomposed question"
    "decomposed_justification_explanation: "The summary explanation for the corresponding decomposed question"
    'pages_info': [
        {"page_name": name of the page included in the search results,
         "page_url": url of the page,
         "page_timestamp": publication timestamp of the page,
         "page_content": scraped content of the page,
         "justification_summary": the page_content summary
        }
        ...
    ]
}
```

## EFFORT Pipeline
Each module below implements a step of EFFORT's pipeline and can be executed separately or as part of the complete workflow:

## Claim Decomp:
To decompopse the claim into a set of sub-questions, you can run module claim_decomposer.py with the following parameters:
--input_path: "string with absolute path to the corpus json dataset"
--output_path: "string with absolute path to the json to store the decomposed dataset"

The following LLM prompt was used:
    "You are a fact-checker. A claim is true when the statement is accurate. A claim is false when the statement is not accurate.
    Take the following claim: '''''claim''''' 
    Assume you will do a web search to verify the claim. What would be the '''''5''''' most important yes or no types of questions to feed a web browser to verify the claim is true and the 
    '''''5''''' most important yes or no types of questions to feed a web browser to verify the claim is false?
    The two sets of 5 questions must explore different aspects of the claim. 
    Return a single list of questions in the following format without any other text: 
    Question: 
    Justification:
    The top five to verify the claim is true and the bottom five to verify the claim is false.

## Evidence Search:
To perform evidence search with the bing API run web_search.py with the following parameters:
--input_path: "string with absolute path to the decomposed json dataset"
--output_path: "string with absolute path to the json to store the web results dataset"
--answer_count: "integer with the target number of urls to return per decomposed question"

The following list was passed to the module as blocked domains:
    "politifact.com",
    "factcheck.org",
    "snopes.com",
    "washingtonpost.com/news/fact-checker/",
    "apnews.com/hub/ap-fact-check",
    "fullfact.org/",
    "reuters.com/fact-check",
    "youtube.com",
    ".pdf",
    "fact-check",
    "factcheck",
    "wikipedia.org",
    "facebook.com"

## Evidence Summarizer:
To perform evidence summarization run justification_summarizer_approach2.py with the following parameters:
--input_path: "string with absolute path to the web search json dataset"
--output_path: "string with absolute path to the json to store the summarized web results dataset"

## Explanation Generator:
To merge the evidence summaries run justification_summaries_merge.py with the following parameters:
--input_path: "string with absolute path to the web summarized web results json dataset"
--output_path: "string with absolute path to the json to store the summary merged dataset"
--FAISS: "1: uses GPT model as in the paper, 0:uses FAISS summarization"

The following LLM prompt was used:
    "Document: '''''All retrieved evidence content'''''

    Summarize the document in a single paragraph. Only include information that is present in the document in a factual manner.
    Your response should not make any reference to "the text" or "the document" and be ready to be merged into a fact-check article."

## Questions Verdicts:
To generate decomposed question verdicts run justifications_classifier.py with the following parameters:
--input_path: "string with absolute path to the summary merge json dataset"
--output_path: "string with absolute path to the json to store the decomposed claim verdict dataset"
--FAISS: "1: uses GPT model as in the paper, 0:uses FAISS summarization"

The following LLM prompt was used:
    "Given the following context: '''''The generated decomposed question explanation'''''.

    Use only the context provided to answer the following question as Yes, No or Unverified.  
    Question: '''''Decomposed question'''''
    Your response should be a single word

## Complete Workflow:
To generate a sample dataset and a CSV test dataset ready for classification run complete_workflow.py. This script automates the modules described above:
--input_path: "string with absolute path to the summary merge json dataset"
--final_output_path: "string with absolute path to the directory that will store all the intermediate json files generated by each module"
--start: "integer number that allows a large dataset to start processing in a row different than 0,      
--end: "integer number that allows a large dataset to end processing in a row different than last,    
--answer_count: "integer with the target number of urls to return per decomposed question"
--use_time_stamp: "1 to use the url timestap in the evidence retrieval module"
--chunk_size: "4 to parallelize and speed up evidence retrieval"
--use_Tavily_search: "this was an experiment with the tavily.ai service for evidence retrieval. Leave it as 0 as it is not used"

## Verdicts Generator:
To generate the final claim verdict, i.e. to run the multi-label classifier model, run model_classifier.py with the following parameters:
--train_file_path: Absolute path to the jsonl file used to train the model. This parameter will only be relevant if "inference" = 0
--test_file_path: Absolute path to the jsonl file used to train the model. This parameter will only be relevant if "inference" = 1
--multi_classifier_path: Absolute path to the pkl model file to be used in the first classifier. See paper for more details
--multi_classifier2_path: Absolute path to the pkl model file to be used in the second classifier. See paper for more details
--binary_classifier_path Absolute path to the pkl model file to a binary model be used in the first classifier. This parameter will only be relevant if "binary_classes" is not set as "skip". See paper for more details
--second_binary_classifier_path: Absolute path to the pkl model file to a binary model be used in the second classifier. This parameter will only be relevant if "second_binary_classes" is not set as "skip". See paper for more details
--SMOTE_type: The type of SMOTE to be used to create synthetic samples to balance the training dataset labels. Options: "minority", "not minority", "not majority", "all"
--model_type: This defines the type of model to be used by model_classifier.py. Supported models: "binary_classifier", regular_multiclassifier", "multiclassifier_except_binaryclass", "two_stage_classifier", "voting_model" 
--model_one_winning_classes: If model type = "voting_model" the winning labels set here will be the ones that will be determined by model 1. The remaining labels will be determined by model 2 
--binary_classes: If model type = "voting_model" or "two_stage_classifier", model 1 binary model will select the predictions that match the labels entered here. If set to "skip", the binary model will be skipped. Example: "true, barely-true" -> the model 1 will take the predicted labels from the binary model that match these labels
--binary_prob_threshold: floating number from 0-1.0. This changes the classifier to assume the label specified in binary_classes if its inference probability is higher tha chosen threshold value. It is only applicable if a single label is defined in binary_classes
--second_binary_classes: Same as binary_classes, but for model 2
--second_binary_prob_threshold: Same as binary_prob_threshold, but for model 2
--five_classes: "0" or "1". If "1" model_classifier.py will modify all the dataset rows where dataset['label']=='pants-fire' to dataset['label']= false
--four_classes: If "1" model_classifier.py will modify all the dataset rows where dataset['label']=='pants-fire' to dataset['label']= false and ['label']=='mostly-true' to dataset['label']= true
--three_classes: If "1" model_classifier.py will modify all the dataset rows where dataset['label']=='pants-fire' to dataset['label']= false and ['label']=='mostly-true' to dataset['label']= true and ['label']=='barely-true' to dataset['label']= half-true
--two_stage_acc_calc: "0" or "1". If "1" terminal will display the intermediate model1 inference report
--feature_engineering: "0" or "1". If "1" will apply feature engineering to the dataset. Currently only Kmeans clustering of features is implemented
--generate_stats: "0" or "1". If "1" will extract the breakdown of all misclassified claims from column dataset['stats_parameter'] (see stats_parameter below)
--stats_parameter: "category" or "subcategory" dataset columns 
--categories_to_remove: categories to remove from the classification report for comparison purposes. It will depend on what is available in the dataset. We testes with "Imagery" and "Not Verifiable"
--plot_charts: "0" or "1". If "1" it will generate plots according to generate_stats
--policy_only: "0" or "1". If "1" model_classifier.py will use only the claims where ['subcategory'] columns is "Politics"
--corpus_file_path: Absolute file path to the corresponding corpus file so the predicted labels are inserted as a new column
--output_file_dir: Absolute directory where the jsonl file including the predicted labels will be saved
--model_label: The label of the model used, given by the user. Just to facilitate analysis later as it is saved to the output dataset
--inference: "0" or "1". If "1" it will run inference, if "0" it will run training

For parameters specific to each model referred to in the paper, check the models directory in the shared folder: https://drive.google.com/drive/folders/1KSni3hrkq8MCyQoOURnEEq7yfyY2qd9T?usp=sharing


## Article Generator and Article Summarizer:
To generate the full verdict explanation article and the corresponding summary run evidence_based_article.py with the following parameters:
--input_path: "string with absolute path to the model classified json dataset"
--output_path: "string with absolute path to the json to store the final dataset including the generated article and summary"
--start: "integer number that allows a large dataset to start processing in a row different than 0,      
--end: "integer number that allows a large dataset to end processing in a row different than last,
--claims_chunk: "integer number that breaks a large dataset into the entered number of rows to perform intermediate file saving"
--output_dir: "directory to store the output dataset"
--scraped_file_path: "string with absolute path to the json dataset including the human-generated article and summary. See scrape_article_summary.py for details"

The following LLM prompt was used:
    "You are a fact-check article writer. Rewrite the following text in the format of an article without a title. '''''The combined decomposed questions explanations'''''
    If the provided text is empty, your should be: "No text provided".
    If not, your answer should return only the article including a conclusion why the following claim is '''''Generated claim verdict''''': '''''Claim'''''

## Other Supporting Script:
The following scripts were used during research and are listed as supporting material to the EFFORT pipeline

## Build Dataset:
build_dataset.py was used to adapt other datasets, in the case of the paper, datacommons.org and https://github.com/sfu-discourse-lab/MisInfoText, to be compatible with ClaimDecomp (used as the baseline dataset). It accepts the following parameters:
--input_path: "absolute path to the csv dataset to be converted",
--output_path: "absolute path to the converted jsonl dataset",
--from_dataset: "if 1, uses input_path, if 0 uses "fact_checker" parameter,
--start: "integer number that allows a large dataset to start processing in a row different than 0,      
--number_of_claims: "integer number that allows a large dataset to end processing in a row different than last,
--fact_checker: "allows a fact-checker website to be scraped. E.g. "politifact.com". It is used if "from_dataset" = 0,
--query_subject: "If scraping a web site, it will filter the types of claims by subject. E.g. "Politics"",
--include_first_category: "1 will fill the "category" column with "query_subject",
--include_subcategory: "1 will fill the "subcategory" column with "query_subject""

## Statistical Analysis:
Run statistical_analysis.py to perform semantic similarity between the generated article/summary and the human generated article/summary, if available. This was done just as a curiosity exercise and is not used in the paper. The following parameters are supported:
--input_path: "absolute path to the final jsonl dataset including generated article/summary and human-generated article/summary,
--output_path: "absolute path to the output jsonl dataset including statitcal metrics",
--corpus_file_path: "the original corpus dataset include the claims under analysis",
--start: "integer number that allows a large dataset to start processing in a row different than 0,      
--end: "integer number that allows a large dataset to end processing in a row different than last,
--columns_to_analyze: "either "article_similarity" or "summary_similarity"",
--label_column: "only "human_label" supported",
--similarity_threshold: "float value used as similarity threshold for comparison",
All parameter belows are "0" or "1" to enable/disable their corresponding analysis
    "--descriptive_stats",
    "--one_sample_t_test",
    "--outlier_detection",
    "--krustal_test",
    "--chi_square_correlation",
    "--visualization",
    "--policy_subcategory",
The parametes below are "0" or "1" depending on the number of labels
    "--five_classes",
    "--four_classes"

## Error Analysis and User Study Data Analysis:
A separate python notebook has been created to each one of the referred tasks. They are stored in corresponding directories of same name.
