from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

dataset = load_dataset("embedding-data/simple-wiki", split="train")


joined_data = [item for inner_list in dataset['set'] for item in inner_list]

maxLength = 32

sentences = []
for s in joined_data:
    words = s.split(" ")
    if len(words) > maxLength:
        words = words[:maxLength]
    if len(words) < maxLength:
        words += ["[PAD]"] * (maxLength - len(words))
    st = " ".join(str(elem) for elem in words)
    sentences.append(st) 

# print(joined_data_correct_length)

word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}  # preset tokens

word_list = list(set(" ".join(sentences).split()))
for i, w in enumerate(word_list):
    word_dict[w] = i + 4  # assign numerical value to every word

vocab_size = len(word_dict)  # number of unique tokens
token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)  # list of everything tokenized

number_dict = {i: w for i, w in
               enumerate(word_dict)}  # get dictionary of all possible numerical representations of words

# print(word_list)


trainSentences = sentences[:int(len(sentences) * 0.8)]
testSentences = sentences[:int(len(sentences) * 0.2)]

def get_training_material():
    return trainSentences, number_dict, word_dict, token_list, vocab_size


def get_testing_material():
    return testSentences, word_dict, token_list 

def get_model_sizes():
    return vocab_size, len(trainSentences)


def get_training_material():
    return trainSentences, number_dict, word_dict, token_list, vocab_size
    

def get_training_vars():
    return len(trainSentences), number_dict


def get_testing_output():
    return testSentences, number_dict 
