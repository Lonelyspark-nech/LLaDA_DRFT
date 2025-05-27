import json
import random
import os
from typing import List, Union

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

E5_EMBEDDING_PROMPTS = {
    "allnli": [
        "Given a premise, retrieve a hypothesis that is entailed by the premise:",
        "Retrieve semantically similar text:",
    ],
    "dureader": "Given a Chinese search query, retrieve web passages that answer the question:",
    "eli5_question_answer": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum:",
    "fever": "Given a claim, retrieve documents that support or refute the claim:",
    "hotpot_qa": "Given a multi-hop question, retrieve documents that can help answer the question:",
    "miracl": "Given a question, retrieve Wikipedia passages that answer the question:",
    "mrtydi": "Given a question, retrieve Wikipedia passages that answer the question:",
    "msmarco_passage": "Given a web search query, retrieve relevant passages that answer the query:",
    "msmarco_document": "Given a web search query, retrieve relevant documents that answer the query:",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question:",
    "quora_duplicates": [
        "Given a question, retrieve questions that are semantically equivalent to the given question:",
        "Find questions that have the same meaning as the input question:",
    ],
    "squad": "Retrieve Wikipedia passages that answer the question:",
    "t2ranking": "Given a Chinese search query, retrieve web passages that answer the question:",
    "trivia_qa": "Retrieve Wikipedia passages that answer the question:",
}


class E5Data(Dataset):
    def __init__(
            self,
            dataset_name: str = "E5",
            split: str = "train",
            file_path: str = "data/echo-data",
            effective_batch_size: int = 32,
            shuffle_individual_datasets: bool = True,
            separator: str = "!@#$%^&*()",
            random_seed: int = 42,  # 新增随机数种子
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.random_seed = random_seed

        self.train_data = []
        self.validation_data = []

        self.load_data(file_path)

    def __len__(self):
        if self.split == "train":
            return len(self.train_data)
        elif self.split == "validation":
            return len(self.validation_data)
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __getitem__(self, index):
        if self.split == "train":
            sample = self.train_data[index]
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            sample = self.validation_data[index]
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def load_data(self, file_path: str = None):
        logger.info(f"Loading E5 data from {file_path}...")

        # 设置随机数种子
        random.seed(self.random_seed)

        data_map = {}
        all_samples = []
        id_ = 0

        for dataset in E5_EMBEDDING_PROMPTS:
            logger.info(f"Loading dataset {dataset}...")
            if dataset not in data_map:
                data_map[dataset] = []
            with open(os.path.join(file_path, f"{dataset}.jsonl"), "r") as f:
                dataset_samples = f.readlines()
            dataset_samples = [json.loads(d) for d in dataset_samples]

            for i, sample in enumerate(dataset_samples):
                instruction = (
                    E5_EMBEDDING_PROMPTS[dataset]
                    if isinstance(E5_EMBEDDING_PROMPTS[dataset], str)
                    else E5_EMBEDDING_PROMPTS[dataset][i % 2]
                )
                query = f"{instruction} " + self.separator + sample["query"] + "<|endoftext|>"
                #pos = self.separator + sample["positive"] + " Use eight words to represent the above text in multiple aspects: <COMBINE01><COMBINE02><COMBINE03><COMBINE04><COMBINE05><COMBINE06><COMBINE07><COMBINE08>"
                #neg = self.separator + sample["negative"] + " Use eight words to represent the above text in multiple aspects: <COMBINE01><COMBINE02><COMBINE03><COMBINE04><COMBINE05><COMBINE06><COMBINE07><COMBINE08>"
                #pos = self.separator + sample["positive"] + " Use eight words to represent the above text in multiple aspects: <|reserved_special_token_0|><|reserved_special_token_1|><|reserved_special_token_2|><|reserved_special_token_3|><|reserved_special_token_4|><|reserved_special_token_5|><|reserved_special_token_6|><|reserved_special_token_7|>"
                #neg = self.separator + sample["negative"] + " Use eight words to represent the above text in multiple aspects: <|reserved_special_token_0|><|reserved_special_token_1|><|reserved_special_token_2|><|reserved_special_token_3|><|reserved_special_token_4|><|reserved_special_token_5|><|reserved_special_token_6|><|reserved_special_token_7|>"
                pos = self.separator + sample["positive"] + "<|endoftext|>"
                neg = self.separator + sample["negative"] + "<|endoftext|>"

                data_map[dataset].append(id_)

                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                    )
                )
                id_ += 1

        # Combine split1 and split2 if any
        new_data_map = {}
        for dataset in data_map:
            new_dataset = dataset.replace("_split1", "").replace("_split2", "")
            if new_dataset not in new_data_map:
                new_data_map[new_dataset] = []
            new_data_map[new_dataset] += data_map[dataset]
        data_map = new_data_map

        if self.shuffle_individual_datasets:
            for task, samples in data_map.items():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        logger.info(
            f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i: i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

        total_len = len(self.data)
        
        split_ratio = 0.05
        num_val = int(total_len * split_ratio)
        random.seed(self.random_seed)
        val_indices = set(random.sample(range(total_len), num_val))
        train_indices = [i for i in range(total_len) if i not in val_indices]

        self.validation_data = [self.data[i] for i in sorted(val_indices)]  
        self.train_data = [self.data[i] for i in train_indices]
        random.shuffle(self.train_data)
        logger.info(f"Split data into {len(self.train_data)} training and {len(self.validation_data)} validation samples.")
