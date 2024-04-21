import urllib.request
import re






url1 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20" \
       "-%20The%20Philosopher's%20Stone.txt "
url2 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%202%20" \
       "-%20The%20Chamber%20of%20Secrets.txt "
url3 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%203%20" \
       "-%20The%20Prisoner%20of%20Azkaban.txt "
# url4 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%204%20" \
#        "-%20The%20Goblet%20of%20Fire.txt "
# url5 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%205%20" \
#        "-%20The%20Order%20of%20the%20Phoenix.txt "
# url6 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%206%20" \
#        "-%20The%20Half%20Blood%20Prince.txt "
# url7 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%207%20" \
#        "-%20The%20Deathly%20Hallows.txt "
# url1 = "https://raw.githubusercontent.com/amephraim/nlp/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%201%20-%20Sorcerer's%20Stone.txt"
# url2 = "https://raw.githubusercontent.com/amephraim/nlp/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%202%20-%20The%20Chamber%20Of%20Secrets.txt"
# url2 = "https://raw.githubusercontent.com/amephraim/nlp/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%203%20-%20Prisoner%20of%20Azkaban.txt"


url8 = "https://raw.githubusercontent.com/amephraim/nlp/master/texts/J.%20K.%20Rowling%20-%20Harry%20Potter%204%20-%20The%20Goblet%20of%20Fire.txt"
# url8 = "https://raw.githubusercontent.com/BrianWeinstein/state-of-the-union/master/transcripts.csv"


def read_file(url, start_phrase, end_phrase, remove):
    """
    :return: formatted text
    """
    with urllib.request.urlopen(url) as webpage:
        readtext = webpage.read().decode("utf8")
        start = readtext.index(start_phrase)
        readtext = readtext[start:]
        end = readtext.index(end_phrase)
        readtext = readtext[:end]
    return readtext.lower().replace('\n', '').replace('¬', '').replace(',', '').replace('.', '').replace(remove, '')


# load in harry potter books

hptext1 = read_file(url1, "Harry Potter and the Sorcerer's Stone", 'THE END', 'Harry Potter and the Philosophers Stone - J.K. Rowling')
hptext2 = read_file(url2, 'HARRY POTTER AND THE CHAMBER OF SECRETS', '*341*', 'Harry Potter and the Chamber of Secrets - J.K. Rowling')
hptext3 = read_file(url3, 'Harry Potter and the Prisoner of Azkaban', 'THE END', 'Harry Potter and the Prisoner of Azkaban')
# hptext3 = read_file(url3, 'OWL POST', 'Page | 487', 'Harry Potter and the Prisoner of Azkaban - J.K. Rowling')
# hptext4 = read_file(url4, 'THE RIDDLE HOUSE', 'Page | 811', 'Harry Potter and the Goblet of Fire - J.K. Rowling')
# hptext5 = read_file(url5, 'DUDLEY DEMENTED', 'Page | 1108', 'Harry Potter and the Order of the Phoenix - J.K. Rowling')
# hptext6 = read_file(url6, 'THE OTHER MINISTER', 'Page | 730', 'Harry Potter and the Half Blood Prince - J.K. Rowling')
# hptext7 = read_file(url7, 'THE DARK LORD ASCENDING', 'Page | 856',
                    # 'Harry Potter and the Deathly Hallows - J.K. Rowling')

# hptext = hptext1 + hptext2 + hptext3 + hptext4 + hptext5 + hptext6 + hptext7
hptext = hptext1 + hptext2 + htptext3
soutext = read_file(url8, 'Harry Potter', 'As Hagrid had said, what would come, would come ... and he would have to meet it when it did.', '')
text = hptext + soutext

# get rid of special characters
bad_ch_list = [';', '#', '$', '%', '&', '@', '[', ']', ' ', ']', '_', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
for ch in bad_ch_list:
    hptext.replace(ch, '')
    soutext.replace(ch, '')

hp_list = re.split('[?.!]', hptext)  # sentence creation
sou_list = re.split('[?.!]', soutext)

sou_list_short = []
for s in sou_list:
    words = s.split(" ")
    if len(words) > 512:
        words = words[:512]
    st = " ".join(str(elem) for elem in words)
    sou_list_short.append(st)  # max 512 words per sentence

hp_list_short = []
for s in hp_list:
    words = s.split(" ")
    if len(words) > 512:
        words = words[:512]
    st = " ".join(str(elem) for elem in words)
    hp_list_short.append(st)  # max 512 words per sentence

sentences = []
for s in hp_list_short:
    sentences.append(s)
word_list = list(set(" ".join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}  # preset tokens
for i, w in enumerate(word_list):
    word_dict[w] = i + 4  # assign numerical value to every word
number_dict = {i: w for i, w in
               enumerate(word_dict)}  # get dictionary of all possible numerical representations of words
vocab_size = len(word_dict)  # number of unique tokens
token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)  # list of everything tokenized
    
testsentences = []
for s in sou_list_short:
    testsentences.append(s)

testword_list = list(set(" ".join(testsentences).split()))  # add to list of possible words
testword_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}  # add list of possible tokens

for i, w in enumerate(testword_list):
    testword_dict[w] = i + 4

testnumber_dict = {i: w for i, w in enumerate(testword_dict)}
testvocab_size = len(testword_dict)  # number of unique tokens
testtoken_list = list()

for sentence in testsentences:
    arr = [testword_dict[s] for s in sentence.split()]
    testtoken_list.append(arr)
    

def get_model_sizes():
    return vocab_size, len(sentences)


def get_training_material():
    return sentences, number_dict, word_dict, token_list, vocab_size
    

def get_testing_material():
    return testsentences, testword_dict, testtoken_list


def get_training_vars():
    return len(sentences), number_dict


def get_testing_output():
    return testsentences, testnumber_dict
