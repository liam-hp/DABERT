from datasets import load_dataset

dataset = load_dataset("embedding-data/simple-wiki", split="train")


joined_data = [item for inner_list in dataset['set'] for item in inner_list]

# new max length of our sentences
# before was 512
maxLength = 32

sentences = []
for s in joined_data:
    words = s.split(" ")
    # truncate if too big
    if len(words) > maxLength:
        words = words[:maxLength]
    # pad if too short
    if len(words) < maxLength:
        words += ["[PAD]"] * (maxLength - len(words))
    st = " ".join(str(elem) for elem in words)
    sentences.append(st) 


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


trainSentences = sentences[:int(len(sentences) * 0.5)]
testSentences = sentences[int(len(sentences) * 0.5):]

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
