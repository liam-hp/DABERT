from transformers import BertTokenizer, DataCollatorForLanguageModeling
import random
import torch
from torch.utils.data import DataLoader, Dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# we will need to customize this

from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import torch

def get_data_loader(sentences, batch_size=32, max_length=32):
    """
    Tokenizes sentences and returns a DataLoader for them.
    :param sentences: List of text sentences.
    :param batch_size: Size of each batch.
    :param max_length: Maximum length of the tokenized output.
    :return: DataLoader with tokenized and appropriately masked batches.
    """
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    
    class SimpleDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    dataset = SimpleDataset(encoded_inputs)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    return dataloader
    
    
# Example usage of the dataset