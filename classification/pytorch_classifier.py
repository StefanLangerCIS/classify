""" Classifier based on pytorch
    Derived from TextClassifier
"""
import itertools
import os
import random
from typing import Any, Dict, Iterable, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import classification.text_classifier
from classification.text_classifier import (ClassifierResult, TextClassifier,
                                            get_data_records_from_file)


class TorchTextClassificationModel(nn.Module):
    """
    Holds the model.
    """

    def __init__(self, vocab_size, embed_dim, num_class):
        """
        Initialize the model with a vocabulary
        :param vocab_size: The size of a vocabulary
        :param embed_dim: The embedding dimensions
        :param num_class: The number of different classes
        """
        super(TorchTextClassificationModel, self).__init__()
        # This layer computes the embedding mean of all tokens
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, mode="mean", sparse=False
        )
        # This layer maps the document embedding to the class labels
        self.linear = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.linear(embedded)


class TorchClassifier(TextClassifier):
    """
    Classify with a pytorch classifier
    """

    def __init__(
        self,
        model_folder_path: str = None,
        embedding_size=64,
        max_training_epochs=8,
        verbose=False,
    ):
        """
        Initialize the classifier.

        :param model_folder_path: The path where to safe the model
        :param verbose: Whether to run in verbose mode with more output
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

        self.verbose = verbose
        self.classifier_name = "pytorch"

        self.embedding_size = embedding_size
        self.learning_rate = 5
        self.max_epochs = max_training_epochs
        self.tokenizer_name = "basic_english"
        self.tokenizer = get_tokenizer(self.tokenizer_name)
        # The torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The following are all determined at training time
        self.vocab = None
        self.text_pipeline = None
        self.label_pipeline = None
        self.model = None
        self.label2number = {}

    def name(self) -> str:
        return self.classifier_name

    def info(self) -> Dict[str, Any]:
        return {"embedding_size": self.embedding_size, "tokenizer": self.tokenizer_name}

    def classify(self, data: dict) -> List[ClassifierResult]:
        """
        Classify a record. The record can have multiple fields with text

        :param data: The data dictionary
        :param text_label: The key(s) in the data which point to the text to classify
        :return: A list with one classifier result
        """
        text = data.get("text", "")
        with torch.no_grad():
            text = torch.tensor(self.text_pipeline(text))
            output = self.model(text, torch.tensor([0]))
            label_number = output.argmax(1).item()
            classifier_result = ClassifierResult(self._get_label_name(label_number))
            return [classifier_result]

    def train(self, training_data: List[Dict]) -> None:
        """
        Train the classifier
        :param training_data: List of training data points with fields 'text' and 'label'
        :return: Nothing
        """
        random.shuffle(training_data)
        split = int(len(training_data) * 0.8)
        data_records_train = training_data[0:split]
        data_records_validate = training_data[split:]
        # Add the "tokens" field to each data record
        self._tokenize(training_data)
        all_tokens = itertools.chain([x["tokens"] for x in training_data])
        self.vocab = build_vocab_from_iterator(all_tokens, specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        # Prepare the text pipeline
        # It is tokenizing the text
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))

        # Provide a mapping from label names to int labels and vice versa
        label_list = [x["label"] for x in training_data]
        for idx, label in enumerate(sorted(set(label_list))):
            self.label2number[label] = idx

        # Add label number to records
        for data_record in training_data:
            label_name = data_record["label"]
            data_record["label_number"] = self.label2number[label_name]

        # Number of different labels
        num_class = len(self.label2number)
        self.model = TorchTextClassificationModel(
            len(self.vocab), self.embedding_size, num_class
        ).to(self.device)
        train_data_loader = DataLoader(
            data_records_train,
            batch_size=8,
            shuffle=False,
            collate_fn=self._collate_batch,
        )
        validation_data_loader = DataLoader(
            data_records_validate,
            batch_size=8,
            shuffle=False,
            collate_fn=self._collate_batch,
        )
        total_accuracy = None
        torch_optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(torch_optimizer, 1.0, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1, self.max_epochs + 1):
            print(f"Starting training for epoch {epoch}")
            self._training_epoch(train_data_loader, criterion, torch_optimizer, epoch)
            accuracy = self._evaluation_epoch(validation_data_loader)
            print(f"Accuracy after epoch {epoch} is {accuracy}")
            if total_accuracy is not None and total_accuracy > accuracy:
                scheduler.step()
            else:
                total_accuracy = accuracy

    def _training_epoch(
        self,
        dataloader: DataLoader,
        criterion: Any,
        torch_optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        """
        Train one epoch.
        :param dataloader: The data
        :param criterion: The loss function
        :param torch_optimizer: The optimizer
        :param epoch: The epoch number
        :return: None
        """
        self.model.train()
        total_acc, total_count = 0, 0
        log_interval = 10

        for idx, (label, text, offsets) in enumerate(dataloader):
            torch_optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            torch_optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                print(
                    f"Training: epoch {epoch} | {idx}/{len(dataloader)} batches | accuracy {(total_acc/total_count):8.3f}"
                )
                total_acc, total_count = 0, 0

    def _evaluation_epoch(self, dataloader: DataLoader):
        """
        One epoch of the evaluation.

        :param dataloader: The evaluation data
        :param criterion:
        :return: The computed accuracy
        """
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = self.model(text, offsets)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count

    def _tokenize(self, data_records: Iterable[Dict]):
        """
        Use the tokenizer to tokenize the records, and add "tokens" field to each
        :param data_records: The records to add the tokens to
        :return: None
        """
        for data_record in data_records:
            data_record["tokens"] = self.tokenizer(data_record["text"])

    def _collate_batch(self, batch):
        """
        Create a batch suitable for the pytorch pipeline
        :param batch: The data in the batch
        :return:
        """
        label_list, text_list, offsets = [], [], [0]
        for data_record in batch:
            label = data_record["label_number"]
            # The original text
            text = data_record["text"]
            label_list.append(label)
            # The tokenized text
            processed_text = torch.tensor(self.text_pipeline(text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return (
            label_list.to(self.device),
            text_list.to(self.device),
            offsets.to(self.device),
        )

    def _get_label_number(self, label_name: str) -> int:
        """
        Map a label name to the number used in pytorch
        :param label_name: The label string
        :return: The label number or -1 if not found
        """
        return self.label2number.get(label_name, -1)

    def _get_label_name(self, label_number: int) -> str:
        """
        Map the number used in the pytorch classifier back to the label name

        :param label_number: The number
        :return: The label name or "no_label" if number is not in the mapping
        """
        for label_name, number in self.label2number.items():
            if number == label_number:
                return label_name
        return "no_label"
