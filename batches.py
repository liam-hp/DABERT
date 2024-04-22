from textblob import TextBlob
import random
# import nltk
# import format_text
import preprocess

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

max_predictions = 32
#sen_length = 512
sen_length = 32 
# sentences, number_dict, word_dict, token_list, vocab_size = format_text.get_training_material()
# testsentences, testword_dict, testtoken_list = format_text.get_testing_material()
sentences, number_dict, word_dict, token_list, vocab_size = preprocess.get_training_material()
testsentences, testword_dict, testtoken_list = preprocess.get_testing_material()

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
        tokens_a = token_list[tokens_a_index]

        actualSentenceLength = len(tokens_a)
        paddingNeeded = sen_length - actualSentenceLength
        traininput_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] 




        # number of predictions we want to make
        num_predictions = min(max_predictions, max(1, int(round(len(
            traininput_ids) * 0.15))))  # grab 15% of words
        #cand_maked_pos = [pos for pos, token in enumerate(traininput_ids) if
        #                  token != word_dict['[CLS]'] and token != word_dict['[SEP]']]

        # list of tokens that we are able to mask
        cand_maked_pos = [pos for pos, token in enumerate(traininput_ids) if
                          token != word_dict['[CLS]'] and token != word_dict['[SEP]']]

        random.shuffle(cand_maked_pos)  # shuffle words
        trainmasked_tokens, trainmasked_pos = [], []

        # for each mod we have to make
        for i in range(0, num_predictions):
            # get the value of the token in the random list of tokens
            token = cand_maked_pos[i]
            # append the token index to the masked pos
            trainmasked_pos.append(token)
            # append the token value
            trainmasked_tokens.append(traininput_ids[token])
            # replace with mask token in input id
            traininput_ids[token] = word_dict['[MASK]']
        
        num_pads = sen_length - len(traininput_ids)
        traininput_ids.extend([0] * num_pads)  # padding input ids
        trainsegment_ids = [0] * (actualSentenceLength) + [1] * (paddingNeeded)

        if max_predictions > num_predictions:  # padding mask
            num_pads = max_predictions - num_predictions
            trainmasked_tokens.extend([0] * num_pads)
            trainmasked_pos.extend([0] * num_pads)
            
        batch.append([traininput_ids, trainsegment_ids, trainmasked_tokens, trainmasked_pos])

    return batch


def make_testing_sentence(rand_index):
    """
    """
    batch = []

    tokens_a_index, tokens_b_index = rand_index, rand_index + 1
    tokens_a, tokens_b = testtoken_list[tokens_a_index], testtoken_list[tokens_b_index]
    testinput_ids = [testword_dict['[CLS]']] + tokens_a + [testword_dict['[SEP]']] + tokens_b + [testword_dict['[SEP]']]
    testsegment_ids = [0] * (sen_length) #+ [1] * (len(tokens_b) + 1)
    testinput_ids = testinput_ids[:sen_length]  # truncated for dimension purposes
    testsegment_ids = testsegment_ids[:sen_length]  # truncated for dimension purposes

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
