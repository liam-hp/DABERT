import math
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import urllib.request
from textblob import TextBlob
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

############################

url1 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20" \
       "-%20The%20Philosopher's%20Stone.txt "
url2 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%202%20" \
       "-%20The%20Chamber%20of%20Secrets.txt "
url3 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%203%20" \
       "-%20The%20Prisoner%20of%20Azkaban.txt "
url4 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%204%20" \
       "-%20The%20Goblet%20of%20Fire.txt "
url5 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%205%20" \
       "-%20The%20Order%20of%20the%20Phoenix.txt "
url6 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%206%20" \
       "-%20The%20Half%20Blood%20Prince.txt "
url7 = "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%207%20" \
       "-%20The%20Deathly%20Hallows.txt "

url8 = "https://raw.githubusercontent.com/BrianWeinstein/state-of-the-union/master/transcripts.csv"


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

hptext1 = read_file(url1, 'THE BOY WHO LIVED', 'Page | 348', 'Harry Potter and the Philosophers Stone - J.K. Rowling')
hptext2 = read_file(url2, 'THE WORST BIRTHDAY', 'Page | 380', 'Harry Potter and the Chamber of Secrets - J.K. Rowling')
hptext3 = read_file(url3, 'OWL POST', 'Page | 487', 'Harry Potter and the Prisoner of Azkaban - J.K. Rowling')
hptext4 = read_file(url4, 'THE RIDDLE HOUSE', 'Page | 811', 'Harry Potter and the Goblet of Fire - J.K. Rowling')
hptext5 = read_file(url5, 'DUDLEY DEMENTED', 'Page | 1108', 'Harry Potter and the Order of the Phoenix - J.K. Rowling')
hptext6 = read_file(url6, 'THE OTHER MINISTER', 'Page | 730', 'Harry Potter and the Half Blood Prince - J.K. Rowling')
hptext7 = read_file(url7, 'THE DARK LORD ASCENDING', 'Page | 856',
                    'Harry Potter and the Deathly Hallows - J.K. Rowling')

hptext = hptext1 + hptext2 + hptext3 + hptext4 + hptext5 + hptext6 + hptext7
soutext = read_file(url8, '2018-01-30', 'and equal government."', 'date,president,title,url,transcript')
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

############################

sen_length = 512  # set sentence length to this for all so equal sentence length
batch_size = len(sentences)
max_predictions = 32
num_layers = 6  # number of Encoder Layer
num_heads = 12  # number of heads in Multi-Head Attention
d_model = 768  # Embedding Size
d_ff = 768 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
dropout = 0.1


def make_training_batch(start_index):
    """
    given start_index, randomly tokenize then mask the words
    :return: a batch of tokens, token locations, masked tokens, and masked token locations
    """
    batch = []

    if start_index + 6 > len(
            sentences):  # may leave off the last few sentences
        end_index = len(sentences) - start_index
    else:
        end_index = start_index + 6  # six sentence pairs

    for idx in range(start_index, end_index):
        tokens_a_index, tokens_b_index = idx, idx + 1
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]  # grab sentence pair
        traininput_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [
            word_dict['[SEP]']]
        trainsegment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        traininput_ids = traininput_ids[:512]  # truncated for dimension purposes
        trainsegment_ids = trainsegment_ids[:512]  # truncated for dimension purposes
        num_predictions = min(max_predictions, max(1, int(round(len(
            traininput_ids) * 0.15))))  # grab 15% of words
        cand_maked_pos = [pos for pos, token in enumerate(traininput_ids) if
                          token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        sampled_cmpos_copy = cand_maked_pos
        sampled_sampled_80 = random.sample(cand_maked_pos[:num_predictions], int(.8 * num_predictions))  # masking 80%
        for rem in sampled_sampled_80:
            sampled_cmpos_copy.remove(rem)
        sampled_sampled_10 = random.sample(sampled_cmpos_copy[:num_predictions], int(
            .1 * num_predictions))  # mark 10% indices
        for rem in sampled_sampled_10:
            sampled_cmpos_copy.remove(rem)
        random.shuffle(cand_maked_pos)  # shuffle words
        trainmasked_tokens, trainmasked_pos = [], []

        for pos in cand_maked_pos[:num_predictions]:
            trainmasked_pos.append(pos)
            trainmasked_tokens.append(traininput_ids[pos])
            if pos in sampled_sampled_80:  # 80% of the marked-to-mask words, as defined above
                traininput_ids[pos] = word_dict['[MASK]']  # make mask
            elif pos in sampled_sampled_10:
                tempindex = random.randint(0, vocab_size - 1)
                traininput_ids[pos] = word_dict[number_dict[tempindex]]  # replace

        num_pads = sen_length - len(traininput_ids)
        traininput_ids.extend([0] * num_pads)  # padding input ids
        trainsegment_ids.extend([0] * num_pads)  # padding segment ids

        if max_predictions > num_predictions:  # padding mask
            num_pads = max_predictions - num_predictions
            trainmasked_tokens.extend([0] * num_pads)
            trainmasked_pos.extend([0] * num_pads)

        batch.append([traininput_ids, trainsegment_ids, trainmasked_tokens, trainmasked_pos])

    return batch


def get_attention_pad_mask(seq_q, seq_k):
    """
    :return: padded attention mask
    """
    batch_size_mask, len_q = seq_q.size()
    batch_size_mask, len_k = seq_k.size()
    pad_attention_mask = seq_k.data.eq(0).unsqueeze(
        1)  # batch_size x 1 x len_k(=len_q), one is masking. eq(0) is PAD token
    return pad_attention_mask.expand(batch_size_mask, len_q, len_k)  # batch_size x len_q x len_k


############################


def tanh(x):
    """
    activation functions
    """
    return (2.0 * torch.special.expit(2.0 * x) - math.sqrt(
        2.0 / math.pi))


def gelu(x):
    """
    activation functions
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0 / math.pi)) * (x + 0.044715 * (x ** 3.0)))


############################

class Embedding(nn.Module):
    """
    does the actual embedding
    """
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(sen_length, d_model)  # position embedding
        self.seg_embed = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    """
    uses attention pad for scores
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    @staticmethod
    def forward(qmat, kmat, vmat, attention_mask):
        scores = torch.matmul(qmat, kmat.transpose(-1, -2)) / np.sqrt(
            d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attention_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attention = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attention, vmat)

        return context, attention


class MultiHeadAttention(nn.Module):
    """
    head of model
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)

    def forward(self, qmat, kmat, vmat, attention_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size_head = qmat, qmat.size(0)

        q_s = self.W_Q(qmat).view(batch_size_head, -1, num_heads, d_k).transpose(1, 2)
        k_s = self.W_K(kmat).view(batch_size_head, -1, num_heads, d_k).transpose(1, 2)
        v_s = self.W_V(vmat).view(batch_size_head, -1, num_heads, d_v).transpose(1, 2)

        attention_mask = attention_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        context, attention = ScaledDotProductAttention()(q_s, k_s, v_s, attention_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size_head, -1, num_heads * d_v)
        output = nn.Linear(num_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attention  # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    """
    forward model
    """
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """
    encoding model
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attention = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attention_mask):
        enc_outputs, attention = self.enc_self_attention(enc_inputs, enc_inputs, enc_inputs,
                                                         enc_self_attention_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attention


class BERT(nn.Module):
    """
    BERT model put together
    """
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = tanh
        self.linear = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        embed_weight = self.embedding.tok_embed.weight
        num_vocab, num_dim = embed_weight.size()
        self.decoder = nn.Linear(num_dim, num_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(num_vocab))

    def forward(self, modelinput_ids, modelsegment_ids, modelmasked_pos):
        output = self.embedding(modelinput_ids, modelsegment_ids)
        enc_self_attention_mask = get_attention_pad_mask(modelinput_ids, modelinput_ids)

        for layer in self.layers:
            output, enc_self_attention = layer(output, enc_self_attention_mask)
        # output : [batch_size, len, d_model], attention : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)

        modelmasked_pos = modelmasked_pos[:, :, None].expand(-1, -1, output.size(-1))  # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, modelmasked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ1(self.norm(self.activ2(self.linear(h_masked))))
        logits_lm_model = self.decoder(h_masked) + self.decoder_bias  # [batch_size, max_pred, n_vocab]

        return logits_lm_model


############################

model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=8e-7)
batch_num = 0

for i in range(0, round(len(sentences) / 6), 128):  # loops through all sentences
    trainingbatch = make_training_batch(i)  # making batch
    input_ids, segment_ids, masked_tokens, masked_pos = map(torch.LongTensor,
                                                            zip(*trainingbatch))
    logits_lm = model(input_ids, segment_ids, masked_pos)
    loss = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
    loss_print = loss / 12  # dividing loss over 12 pairs
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    batch_num += 1
    print(f'Batch Number: {batch_num} | Loss: {loss_print:.4f}')

############################

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


def make_testing_sentence(rand_index):
    """
    grabs testing sentence for State of the Union address, masks all nouns to be replaces with Harry Potter nouns
    :return: test batch, similar to training batch
    """
    batch = []

    tokens_a_index, tokens_b_index = rand_index, rand_index + 1
    tokens_a, tokens_b = testtoken_list[tokens_a_index], testtoken_list[tokens_b_index]
    testinput_ids = [testword_dict['[CLS]']] + tokens_a + [testword_dict['[SEP]']] + tokens_b + [testword_dict['[SEP]']]
    testsegment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
    testinput_ids = testinput_ids[:512]  # truncated for dimension purposes
    testsegment_ids = testsegment_ids[:512]  # truncated for dimension purposes

    num_predictions = min(max_predictions, max(1, int(round(len(testinput_ids) * 0.15))))  # cap predictions
    cand_maked_pos = [pos for pos, token in enumerate(testinput_ids) if
                      token != word_dict['[CLS]'] and token != word_dict['[SEP]']]  # grab copy of list
    testmasked_tokens, testmasked_pos = [], []  # create mask tokens, positions list

    # grabbing the nouns
    s_with_nouns = TextBlob(testsentences[rand_index] + testsentences[rand_index + 1])
    # noinspection PyTypeChecker
    nouns = list(s_with_nouns.tags)
    nouns_to_grab = []

    for pos in range(len(nouns)):
        word, type_of_word = nouns[pos]
        if type_of_word == 'NNP' or type_of_word == 'NN':
            if pos < len(cand_maked_pos):
                nouns_to_grab.append(cand_maked_pos[pos])
    for pos in nouns_to_grab:
        testmasked_pos.append(pos)  # add position
        testmasked_tokens.append(testinput_ids[pos])  # add token
        testinput_ids[pos] = testword_dict['[MASK]']  # make mask

    num_pads = sen_length - len(testinput_ids)
    testinput_ids.extend([0] * num_pads)  # padding input ids
    testsegment_ids.extend([0] * num_pads)  # padding segment ids

    if max_predictions > num_predictions:  # padding mask
        num_pads = max_predictions - num_predictions
        testmasked_tokens.extend([0] * num_pads)
        testmasked_pos.extend([0] * num_pads)

    batch.append([testinput_ids, testsegment_ids, testmasked_tokens, testmasked_pos])

    return batch


############################

index = random.randrange(len(testsentences) - 1)
old_sentence = testsentences[index] + testsentences[index + 1]
printinput_ids, printsegment_ids, printmasked_tokens, printmasked_pos = map(torch.LongTensor,
                                                                            zip(*make_testing_sentence(index)))

masked_sentence = " ".join(
    [testnumber_dict[w.item()] for w in printinput_ids[0] if testnumber_dict[w.item()] != '[PAD]'])
print('masked sentence : ', masked_sentence)  # sentence with mask still in
print('actual sentence : ', old_sentence)  # sentence without mask
print('masked tokens list : ',
      [pos.item() for pos in printmasked_tokens[0] if pos.item() != 0])  # tokens of words covered by mask

logits_lm = model(printinput_ids, printsegment_ids, printmasked_pos)
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])  # tokens of predicted noun

masked_token_list = [pos.item() for pos in printmasked_tokens[0] if pos.item() != 0]
predicted_mt_list = [pos for pos in logits_lm if pos != 0]

list_for_output = []
maskpos = 0
for w in printinput_ids[0]:
    if testnumber_dict[w.item()] != '[PAD]':  # if not padded
        if testnumber_dict[w.item()] == '[MASK]':  # if word got randomly masked
            list_for_output.append(number_dict[predicted_mt_list[maskpos].item()])
            maskpos += 1
        else:  # if word not masked
            list_for_output.append(testnumber_dict[w.item()])

newsentence = " ".join(list_for_output)
print(newsentence)  # sentence with Harry Potter noun
