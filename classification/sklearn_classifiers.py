""" Classifiers based on the sklearn
    They all have the same interface, so the can be wrapped in one class
    Derived from TextClassifier
"""
import random
import re
from typing import List, Dict, Any

import numpy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD

from text_encoder import TextEncoder
from text_classifier import TextClassifier, ClassifierResult

import json
import os
import pandas

# Sklearn: Classifiers
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

# Sklearn: Other utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC


class SklearnClassifier(TextClassifier):
    """
    Classify with sklearn classifiers
    """
    supported_classifiers = ["DecisionTreeClassifier", "RandomForestClassifier", "LogisticRegression", "MLPClassifier",
                             "GaussianNB", "MultinomialNB", "KNeighborsClassifier", "LinearSVC", "Perceptron"]

    def __init__(self, classifier_type: str, dense: bool = False, lsa: bool = False, model_folder_path: str = None, verbose=False):
        """
        Initialize the classifier
        :param classifier_type: The name of the classifiers
        :param dense: Produce a dense vector (using transformers)
        :param lsa: Produce a dense vector using latent semantic analysis / singular value decomposition
        :param model_folder_path:
        :param verbose:
        """
        if model_folder_path:
            self.model_folder_path = model_folder_path
        else:
            model_path = os.path.abspath(__file__)
            model_path = os.path.dirname(model_path)
            model_path = os.path.join(model_path, "data", "models")
            self.model_folder_path = model_path
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        # Store the file path of the training data
        self.training_data = None
        self.verbose = verbose

        if classifier_type == "KNeighborsClassifier":
            self.sklearn_classifier = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
        elif classifier_type == "MLPClassifier":
            self.sklearn_classifier = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic')
        elif classifier_type == "LinearSVC":
            self.sklearn_classifier = LinearSVC()
        elif classifier_type == "GaussianNB":
            self.sklearn_classifier = GaussianNB()
        elif classifier_type == "MultinomialNB":
            self.sklearn_classifier = MultinomialNB(alpha=0.01)
        elif classifier_type == "LogisticRegression":
            self.sklearn_classifier = LogisticRegression()
        elif classifier_type == "RandomForestClassifier":
            self.sklearn_classifier = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        elif classifier_type == "DecisionTreeClassifier":
            self.sklearn_classifier = DecisionTreeClassifier(min_samples_split=10, max_features='sqrt')
        elif classifier_type == "Perceptron":
            self.sklearn_classifier = Perceptron()
        else:
            raise Exception(
                "Unsupported classifier type {0}. Use one of {1}".format(classifier_type, self.supported_classifiers))

        self.classifier_type = classifier_type
        self.dense = dense
        # Do a feature reduction using latent semantic analysis
        self.lsa = lsa
        self.classifier_name = classifier_type
        if self.dense:
            self.classifier_name = self.classifier_name + "-dense"
        if self.lsa:
            self.classifier_name = self.classifier_name + "-lsa"
        self.count_vectorizer = None
        self.tfidf_transformer = None
        self.svd_transformer = None
        self.sentence_transformer = None

    def name(self) -> str:
        return self.classifier_name

    def info(self) -> Dict[str, Any]:
        return self.sklearn_classifier.get_params()

    def classify(self, data: dict, text_label: List[str]) -> List[ClassifierResult]:
        """
        Classify a record consisting of text and sensor codes
        :return The detected class as ClassifierResult
        """
        # print(sensor_codes)
        data_point = {}
        data_point["text"] = " ".join([data[x] for x in text_label])
        data_table = create_data_table([data_point])
        if self.dense:
            matrix = self._create_dense_matrix(data_table)
        else:
            matrix = self._create_tfidf_matrix(data_table)

        predicted = self.sklearn_classifier.predict(matrix)
        predicted_class = predicted[0]

        result = ClassifierResult(predicted_class, -1, "")
        return [result]

    def train(self, training_data: str, text_label: List[str], class_label: str) -> None:
        """
        Train the classifier
        :param training_data: File name. Training data is one json per line
        :param text_label: Json field which contains the text
        :param class_label:  Json field which contains the label for the classes to train
        :return: Nothing
        """
        """
        Train the algorithm with the data from the knowledge graph
        """
        self.training_data = training_data
        data_points = get_data_records_from_file(training_data, text_label, class_label)
        data_train = create_data_table(data_points)
        # data_train = data_train.truncate(after=200)
        if self.dense:
            matrix_train = self._create_dense_matrix(data_train)
        else:
            matrix_train = self._train_tfidf_matrix(data_train)
        print("INFO: Starting fitting of classifier")
        self.sklearn_classifier.fit(matrix_train, data_train.label)
        print("INFO: Fitting completed")
        self._save_model_information()

    def _save_model_information(self) -> None:
        """
        Print detailed information about the model
        :return: None
        """
        pass
        # if self.classifier_type == "DecisionTreeClassifier":
        #    self.print_decision_tree()
        #else:
        #    pass

    def _print_decision_tree(self):
        """
        Print decision tree rules
        :return: 
        """
        rules_text = export_text(self.sklearn_classifier, max_depth=100)
        # Vocabulary for replacement in the data which contains
        # feature numbers only
        vocab = self.count_vectorizer.vocabulary_
        vocabulary = dict((feature, word) for word, feature in vocab.items())
        rules = rules_text.split("\n")
        lines = []
        for rule in rules:
            if "feature_" in rule:
                word_id_str = re.sub(".*feature_([0-9]+).*", r"\1", rule)
                word_id = int(word_id_str)
                if word_id in vocabulary:
                    word = vocabulary[word_id]
                else:
                    word = "UNK"
                rule = rule.replace("feature_{}".format(word_id_str), word)
                lines.append(rule)
            else:
                lines.append(rule)

        with open(os.path.join(self.model_folder_path, "decision_rules.txt"), 'w', encoding='utf-8') as out:
            for line in lines:
                out.write(line + '\n')

    def _train_tfidf_matrix(self, data_train: pd.DataFrame) -> numpy.ndarray:
        """
        Based on a pandas data frame, build the TF-IDF table
        :param data_train: Data frame containing the text in column 'text'
        :return: The TF-IDF term count matrix as numpy array
        """
        print("INFO: creating TF-IDF Matrix")
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(min_df=2, max_df=0.8, ngram_range=(1, 1))
        matrix_train_counts = self.count_vectorizer.fit_transform(data_train.text)
        if self.tfidf_transformer is None:
            self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(matrix_train_counts)
        matrix_train = self.tfidf_transformer.transform(matrix_train_counts)
        print("INFO: TF-IDF matrix creation completed")
        if self.lsa:
            print("INFO: Reducing matrix with LSA")
            self.svd_transformer = TruncatedSVD(n_components=200, random_state=0)
            matrix_train = self.svd_transformer.fit_transform(matrix_train)
            print("INFO: LSA matrix reduction completed")
        else:
            matrix_train = matrix_train.toarray()

        return matrix_train

    def _create_tfidf_matrix(self, data_to_classify: pd.DataFrame) -> numpy.ndarray:
        matrix_counts = self.count_vectorizer.transform(data_to_classify.text)
        matrix = self.tfidf_transformer.transform(matrix_counts)
        if self.lsa:
            matrix = self.svd_transformer.transform(matrix)
        else:
            matrix = matrix.toarray()
        return matrix

    def _create_dense_matrix(self, data: pd.DataFrame) -> numpy.ndarray:
        """
        Create a dense matrix based on some deep learning model
        :param data: The data to create the dense matrix from
        :return: 
        """""
        if self.sentence_transformer is None:
            embeddings_cache = os.path.join(self.model_folder_path, "embeddings.jsonl")
            embeddings_cache_english = os.path.join(self.model_folder_path, "embedding_english.jsonl")
            # Model selected for good speed and ok-ish performance
            # see https://www.sbert.net/docs/pretrained_models.html
            # Multilingual
            # self.sentence_transformer = TextEncoder('paraphrase-multilingual-MiniLM-L12-v2', cache_file=embeddings_cache)
            # Monolingual
            self.sentence_transformer = TextEncoder('all-MiniLM-L12-v2', cache_file=embeddings_cache_english)

        encoded = self.sentence_transformer.encode(data.text)
        return encoded


# ********************************
# Creation  of data to classify
# ********************************
def get_data_records_from_file(training_file: str, text_label: List[str], class_label: str, mx: int = 0):
    """
    Retrieve the data records from file (for training)
    """
    with open(training_file, encoding='utf-8') as training_fp:
        data_records = []
        for line in training_fp:
            record = json.loads(line)
            data_record = {}
            data_record["text"] = " ".join([record[x] for x in text_label])
            data_record["label"] = record[class_label]
            data_records.append(data_record)

    if mx is not None and mx > 0:
        data_records = random.sample(data_records, mx)

    return data_records


def create_data_table(data_records: List) -> pandas.DataFrame:
    """
    Create a shuffled data frame
    :param data_records: The data records to create the data table from
    :return: The panda data frame
    """
    data_records = shuffle(data_records)
    data_table = pandas.DataFrame(data_records)
    data_table.fillna(0, inplace=True)
    return data_table
