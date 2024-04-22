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

    if start_index + 64 > len(sentences):  # may leave off the last few sentences
        end_index = len(sentences) - start_index
    else:
        end_index = start_index + 64  # six sentence pairs

    for idx in range(start_index, end_index):
        tokens_a_index, tokens_b_index = idx, idx + 1
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]  # grab sentence pair
        # print("len", len(tokens_a), len(tokens_b))
        traininput_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [
            word_dict['[SEP]']]
        trainsegment_ids = [0] * (sen_length)# + [1] * (len(tokens_b) + 1)
        traininput_ids = traininput_ids[:sen_length]  # truncated for dimension purposes
        trainsegment_ids = trainsegment_ids[:]  # truncated for dimension purposes
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

    # print("training batch", batch)
    # for i in range(len(batch)):
    #     print("training batch", batch[i])
    #     for j in range(len(batch[i])):
    #         print(batch[i][j])
    #         print("\tshape:", len(batch[i][j]))


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
