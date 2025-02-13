import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from itertools import groupby
import nltk
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import time
from tabulate import tabulate

# Check if the punkt_tab file exists, if not, download it
nltk.download('punkt_tab')

# Vocabulary of 2000 values because there are 2000 clusters
DSU_VOCABULARY = [str(i) for i in range(2001)]

COLUMNS = ['text', 'DSU', 'DSU_deduplicated']

# We suggest that local folder is Speaker-recognition
#GLOBAL_PATH = '/home/liskasi/Projects/Speaker-recognition'
#GLOBAL_PATH = Path(os.path.expanduser("~"))

current_file_path = Path(__file__).resolve()
# Navigate up to the repository root directory (assuming the repository structure is consistent)
REPO_ROOT = current_file_path.parent.parent
VCTK_DATASET = 'VCTK'
LS_DATASET = 'LibriSpeech'
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SGD SVM": SGDClassifier(loss='hinge'),
    "SGD LogReg": SGDClassifier(loss='log_loss'),
}

def load_data(dataset_name):
    """
    Load training and test datasets for a given dataset name.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'VCTK' or 'LibriSpeech')

    Returns:
        tuple: Training and test DataFrames for the specified dataset
    """
    train_path = Path(REPO_ROOT, 'DSU_creation', dataset_name, 'final_data_' + dataset_name.lower(), 'train.csv')
    test_path = Path(REPO_ROOT, 'DSU_creation', dataset_name, 'final_data_' + dataset_name.lower(), 'test.csv')
    test_data = pd.read_csv(test_path)
    train_data = pd.read_csv(train_path)

    return train_data, test_data


def add_dsu_deduplicated_column(df):
    """
    Create a new column with deduplicated DSU tokens.
    
    Args:
        df (DataFrame): Input DataFrame
    
    Returns:
        DataFrame: DataFrame with added 'DSU_deduplicated' column
    """
    df[f'DSU_deduplicated'] = df['DSU'].apply(
        lambda x: ' '.join([k for k, _ in groupby(x.split())])
    )
    return df


def prepare_data():
    """
    Prepare datasets by loading VCTK and LibriSpeech data and adding deduplicated columns.
    
    Returns:
        list: List of processed datasets (train and test for both VCTK and LibriSpeech)
    """
    vctk_train_data, vctk_test_data = load_data(VCTK_DATASET) # Data for VCTK dataset
    ls_train_data, ls_test_data = load_data(LS_DATASET) # Data for LibriSpeech dataset

    datasets = [vctk_train_data, vctk_test_data, ls_train_data, ls_test_data]

    # Adding DSU_deduplicated column for each dataset
    datasets_deduplicated = [add_dsu_deduplicated_column(dataset) for dataset in datasets]
    return datasets_deduplicated


def tokenize_dsu(dsu_string):
    """
    Tokenize DSU string by splitting on whitespace.
    
    Args:
        dsu_string (str): Input DSU string
    
    Returns:
        list: List of tokens
    """
    return dsu_string.split()
    

def extract_bigrams(data, column_name):
    """
    Extracts unique bigrams from a given dataset.

    Args:
        data (dict): A dictionary containing a key 'DSU_deduplicated', which maps
                      to a list of text entries. Each entry is a string of text
                      where words are separated by spaces.

    Returns:
        list: A list of unique bigrams found in the 'DSU_deduplicated' texts.
    """
    
    all_bigrams = set()
    for text in data[column_name]:
        tokens = text.split()
        bigrams = [" ".join(pair) for pair in zip(tokens, tokens[1:])]
        all_bigrams.update(bigrams)

    return list(all_bigrams)


def vectorize_data(vectorizer, train_data, test_data, ngram_range):
    """
    Vectorizes columns using the specified vectorizer class.

    Args:
        vectorizer (class): Vectorization method (e.g., CountVectorizer or TfidfVectorizer)
        train_data (DataFrame): Training dataset
        test_data (DataFrame): Test dataset
        ngram_range (tuple): Range of n-grams to consider.

    Returns:
        dict: Dictionary containing vectorized features.
    """
    vectorized_features = {}

    for column_name in COLUMNS:
        tokenizer = word_tokenize if column_name == 'text' else tokenize_dsu
        if column_name == 'text': vocabulary = None
        elif ngram_range == (1, 1): vocabulary = DSU_VOCABULARY
        elif ngram_range == (2, 2): vocabulary = extract_bigrams(train_data, column_name)
        else: raise ValueError("Invalid ngram_range")

        v = vectorizer(tokenizer=tokenizer, vocabulary=vocabulary, ngram_range=ngram_range)

        v.fit(train_data[column_name])
        vectorized_features[column_name] = v.transform(test_data[column_name])

    return vectorized_features


def create_vectorized_datasets(vectorizer, datasets, ngram_range=(1, 1)):
    """
    A dictionary containing preprocessed datasets for speaker recognition, including both 
    training and testing data for two datasets: LibriSpeech and VCTK.

    Args:
        vectorizer (class): For example, CountVectorizer or TfidfVectorizer.
        datasets (array): Array containing the training and test datasets.
        ngram_range (tuple, optional): Range of n-grams to consider. Default is (1, 1).

    Returns:
        dict: Dictionary containing preprocessed datasets for speaker recognition.
    """

    vctk_train_data, vctk_test_data, ls_train_data, ls_test_data = datasets

    return {
    'librispeech': {
        'train': {
            'features': vectorize_data(vectorizer, ls_train_data, ls_train_data, ngram_range=ngram_range),
            'labels': ls_train_data['speaker_id']
        },
        'test': {
            'features': vectorize_data(vectorizer, ls_train_data, ls_test_data, ngram_range=ngram_range),
            'labels': ls_test_data['speaker_id']
        }
    },
    'vctk': {
        'train': {
            'features': vectorize_data(vectorizer, vctk_train_data, vctk_train_data, ngram_range=ngram_range),
            'labels': vctk_train_data['speaker_id']
        },
        'test': {
            'features': vectorize_data(vectorizer, vctk_train_data, vctk_test_data, ngram_range=ngram_range),
            'labels': vctk_test_data['speaker_id']
        }
    }
}

def visualize_pca(datasets, top_n_speakers=5, n_grams=(1, 1)):
    """
    Visualize datasets using Principal Component Analysis (PCA).
    
    Args:
        datasets (dict): Processed datasets
        top_n_speakers (int, optional): Number of top speakers to visualize. Defaults to 5.
        n_grams (tuple, optional): Range of n-grams. Defaults to (1, 1).
    """

    for dataset_name, dataset_data in datasets.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"PCA Visualization - {dataset_name.capitalize()} (n_grams = {n_grams})", fontsize=16)
        palette = sns.color_palette("colorblind", n_colors=top_n_speakers)

        for i, feature_name in enumerate(COLUMNS):
            X_train = dataset_data['train']['features'][feature_name]
            y_train = dataset_data['train']['labels']

            pca = PCA(n_components=2)
            X_train_pca = pca.fit_transform(X_train.toarray())

            pca_df = pd.DataFrame(data=X_train_pca, columns=['PC1', 'PC2'])
            pca_df['speaker_id'] = y_train

            top_speakers = pca_df['speaker_id'].value_counts().nlargest(top_n_speakers).index

            pca_df_filtered = pca_df[pca_df['speaker_id'].isin(top_speakers)]

            sns.scatterplot(x='PC1', y='PC2', hue='speaker_id', data=pca_df_filtered, ax=axes[i], palette=palette)
            axes[i].set_title(feature_name.capitalize())
            axes[i].get_legend()
            axes[i].grid()

        plt.tight_layout()
        plt.show()


def visualize_tsne(datasets, top_n_speakers=5, n_grams=(1, 1)):
    """
    Visualize datasets using t-SNE.
    
    Args:
        datasets (dict): Processed datasets
        top_n_speakers (int, optional): Number of top speakers to visualize. Defaults to 5.
        n_grams (tuple, optional): Range of n-grams. Defaults to (1, 1).
    """

    for dataset_name, dataset_data in datasets.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"t-SNE Visualization - {dataset_name.capitalize()} (n_grams = {n_grams})", fontsize=16)
        palette = sns.color_palette("colorblind", n_colors=top_n_speakers)

        for i, feature_name in enumerate(COLUMNS):
            X_train = dataset_data['train']['features'][feature_name]
            y_train = dataset_data['train']['labels']

            tsne = TSNE(n_components=2)
            X_train_tsne = tsne.fit_transform(X_train.toarray())

            tsne_df = pd.DataFrame(data=X_train_tsne, columns=['t-SNE1', 't-SNE2'])
            tsne_df['speaker_id'] = y_train

            top_speakers = tsne_df['speaker_id'].value_counts().nlargest(top_n_speakers).index

            tsne_df_filtered = tsne_df[tsne_df['speaker_id'].isin(top_speakers)]

            sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='speaker_id', data=tsne_df_filtered, ax=axes[i], palette=palette)
            axes[i].set_title(feature_name.capitalize())
            axes[i].get_legend()
            axes[i].grid()

        plt.tight_layout()
        plt.show()


def train_vanilla_models(datasets):
    """
    Train multiple vanilla classifiers on different datasets and feature types.
    
    Args:
        datasets (dict): Processed datasets
    
    Returns:
        DataFrame: Performance metrics for each model configuration
    """
    results = []

    with tqdm(total=len(datasets) * len(COLUMNS) * len(CLASSIFIERS), desc="Overall Training Progress") as pbar:
            for dataset_name, dataset_data in datasets.items():
                for feature_name in COLUMNS:
                    # Get training data
                    X_train = dataset_data['train']['features'][feature_name]
                    y_train = dataset_data['train']['labels']

                    # Get test data
                    X_test = dataset_data['test']['features'][feature_name]
                    y_test = dataset_data['test']['labels']

                    for model_name, model in CLASSIFIERS.items():
                        pbar.set_description(f"Training: {dataset_name} - {feature_name} - {model_name}")
                        
                        # Train the model
                        start_time = time.time()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                        end_time = time.time()

                        results.append({
                            "Dataset": dataset_name,
                            "Feature": feature_name,
                            "Model": model_name,
                            "Accuracy": accuracy_score(y_test, predictions),
                            "Precision": precision_score(y_test, predictions, average='weighted'),
                            "Recall": recall_score(y_test, predictions, average='weighted'),
                            "F1-Score": f1_score(y_test, predictions, average='weighted'),
                            "Execution_Time": end_time - start_time,
                        })

                        pbar.update(1)
    return pd.DataFrame(results)


def visualise_training_results_per_feature(results_df):
    """
    Visualize model training results with tabular and bar chart representations.
    
    Args:
        results (DataFrame): Performance metrics from model training
    """
    # Group results by feature type
    results_df['Feature'] = pd.Categorical(results_df['Feature'], categories=COLUMNS, ordered=True)
    grouped_results = results_df.groupby('Feature')

    for column_name, group in grouped_results:
        print(f"\nResults for type: {column_name}")
        print(tabulate(group.drop(columns=["Feature"]), headers='keys', tablefmt='psql', showindex=False))
        
        plt.figure(figsize=(10, 6))
        sns.set_palette("colorblind")
        sns.barplot(x='Model', y='F1-Score', data=group, hue='Dataset')
        plt.title(f"Performance of classifiers for {column_name} Feature")
        plt.xlabel('Model')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



def visualize_training_results_for_dataset(results_df, dataset_name, metric, n_grams='unigrams'):
    """
    Visualizes results for a specific dataset and metric with features as different colored bars.

    Args:
        results_df (DataFrame): The DataFrame containing the results.
        dataset_name (str): The name of the dataset to visualize.
        metric (str): The metric to plot on the y-axis.
        n_grams (str, optional): Range of n-grams. Defaults to 'unigrams'.
    """
    dataset_results = results_df[results_df['Dataset'] == dataset_name]

    if dataset_results.empty:
        print(f"No results found for dataset: {dataset_name}")
        return

    try:
        plt.figure(figsize=(6, 6))
        sns.set_palette("colorblind")  # Use a visually distinct color palette
        ax = sns.barplot(x='Model', y=metric, data=dataset_results, hue='Feature')  # Сохраняем axes

        plt.title(f"Classifiers for {dataset_name} Dataset ({metric}) ({n_grams})")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        if metric != 'Execution_Time':
            ax.set_ylim(0, 1)  # или plt.ylim(0, 1)

        plt.show()

    except KeyError as e:
        print(f"Error: Column '{metric}' not found in the results DataFrame")
        print(f"Available columns: {dataset_results.columns.tolist()}")


def visualise_training_results_per_dataset(results_df, metric='F1-Score', n_grams='unigrams', tables=True, bar_charts=True):
    """
    Visualize model training results with tabular and bar chart representations.
    Calls the new function to create bar charts per dataset.

    Args:
        results_df (DataFrame): The DataFrame containing the results.
        metric (str, optional): The metric to plot on the y-axis. 'F1-Score' is used by default.
        n_grams (str, optional): Range of n-grams. Defaults to 'unigrams'.
        tables (bool, optional): Whether to display tables. Defaults to True.
        bar_charts (bool, optional): Whether to display bar charts. Defaults to True.
    """
    results_df['Feature'] = pd.Categorical(results_df['Feature'], categories=COLUMNS, ordered=True)
    grouped_results = results_df.groupby('Feature')

    if tables:
        for column_name, group in grouped_results:
            print(f"\nResults for type: {column_name} (n_grams = {n_grams})")
            print(tabulate(group.drop(columns=["Feature"]), headers='keys', tablefmt='psql', showindex=False))

    if bar_charts:
        for dataset_name in results_df['Dataset'].unique():
            visualize_training_results_for_dataset(results_df, dataset_name, metric=metric, n_grams=n_grams)

    
def train_grid_search(datasets, dataset_name, feature_name, model, param_grid, scoring_metric='f1_weighted'):
    """
    Perform grid search using GridSearchCV from sklearn on a single feature.

    Args:
        datasets (dict): Processed datasets.
        dataset_name (str): Name of the dataset to use.
        feature_name (str): Name of the feature to use.
        model (object): The model to train.
        param_grid (dict): Dictionary of parameters to search.
        results_df (DataFrame, optional): DataFrame to store new results.
        scoring_metric (str, optional): The metric to use for scoring. Default is 'f1_weighted'.

    Returns:
        dict: A dictionary containing accuracy, F1-score, and confusion matrix 
              for the best performing model on the test set.
    """

    best_params = {}
    best_model = None

    X_train = datasets[dataset_name]['train']['features'][feature_name]
    y_train = datasets[dataset_name]['train']['labels']
    X_test = datasets[dataset_name]['test']['features'][feature_name]
    y_test = datasets[dataset_name]['test']['labels']

    print(f"Starting grid search for {dataset_name} - {feature_name} - {model.__class__.__name__}")

    grid_search = GridSearchCV(model, param_grid, scoring=scoring_metric, cv=5, verbose=5, n_jobs=1)  # verbose=3 для подробного вывода

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_


    print(f"Best parameters for {dataset_name} - {feature_name} - {model.__class__.__name__}: {best_params}")

    predictions = best_model.predict(X_test)

    results_df = [{
                        "Dataset": dataset_name,
                        "Feature": feature_name,
                        "Model": model.__class__.__name__,
                        "Accuracy": accuracy_score(y_test, predictions),
                        "F1-Score": f1_score(y_test, predictions, average='weighted'),
    }]
    conf_matrix = [confusion_matrix(y_test, predictions)]

    plt.figure(figsize=(20, 15))
    sns.heatmap(conf_matrix[0], annot=True, fmt="d", cmap="Blues")
    plt.title(f"Best parameters predictions: {dataset_name} - {feature_name} - {model.__class__.__name__}")
    plt.show()

    return results_df, conf_matrix