import random
import torch
from torch import cuda
import torch.optim as optim
import torch.nn as nn
import model_architecture
# import early_stopping
import betterbatches
from torch.utils.data import DataLoader
# import format_text
import test
from transformers import BertConfig

import preprocess

device = "cuda" if cuda.is_available() else "cpu"

model_architecture
preprocess
test



def run_bert(attention_type):
    sentences, number_dict, word_dict, token_list, vocab_size = preprocess.get_training_material()
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = test.CustomBertModel(config).to(device) # Single Linear Layer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=8e-7)
    databatches = betterbatches.get_data_loader(sentences)
        # early_stopping = early_stopping.EarlyStopping(tolerance=20, min_delta=0.01) # early stop when 10 occurances where loss does not decrease by > 0.01

    length_sentences, number_dict = preprocess.get_training_vars()
    testsentences, testnumber_dict = preprocess.get_testing_output()

    losses = []

    print(f"Initialization complete. Training on {length_sentences} example sentences.")

    sentences, number_dict, word_dict, token_list, vocab_size = preprocess.get_training_material()

    for batch_num, batch in enumerate(databatches):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to the appropriate device
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        batchloss = loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Batch Number: {batch_num} | Loss: {batchloss:.4f}')
        losses.append(batchloss.item())

    averageLast5 = losses[-5:]
    average = sum(averageLast5) / 5
    # print("average last 5 losses:", average)
    return average

    # batch_count = round(length_sentences / 6)
    # best_loss = 100
    # for i in range(0, batch_count, 128):  # loops through all sentences
    #     trainingbatch = batches.make_training_batch(i)  # making batch
    #     input_ids, segment_ids, masked_tokens, masked_pos = map(torch.LongTensor,
    #                                                             zip(*trainingbatch))
    #     input_ids = input_ids.to(device)
    #     segment_ids = segment_ids.to(device)
    #     masked_tokens = masked_tokens.to(device)
    #     masked_pos = masked_pos.to(device)

    #     # print("batch input", input_ids)
    #     # print("batch masked", masked_tokens)
    #     # print("batch madked pos", masked_pos)
    #     logits_lm = model(input_ids, segment_ids, masked_pos)
    #     loss = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
    #     loss_print = loss / 12  # dividing loss over 12 pairs
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     batch_num += 1
    #     print(f'Batch Number: {batch_num} | Loss: {loss_print:.4f}')
    #     losses.append(loss_print.item())
    #     # if early_stopping(loss_print.item()):
    #     #     print("Early stopping at epoch:", i, "batch:", batch_num)
    #     #     break

    #     # index = random.randrange(len(testsentences) - 1)
    #     # old_sentence = testsentences[index] + testsentences[index + 1]
    #     # printinput_ids, printsegment_ids, printmasked_tokens, printmasked_pos = map(torch.LongTensor,
    #     #                                                                             zip(*batches.make_testing_sentence(index)))

    #     # printinput_ids = printinput_ids.to(device)
    #     # printsegment_ids = printsegment_ids.to(device)
    #     # printmasked_tokens = printmasked_tokens.to(device)
    #     # printmasked_pos = printmasked_pos.to(device)
    #     # masked_sentence = " ".join(
    #     #     [testnumber_dict[w.item()] for w in printinput_ids[0] if testnumber_dict[w.item()] != '[PAD]'])
    #     # # print('masked sentence : ', masked_sentence)  # sentence with mask still in
    #     # # print('actual sentence : ', old_sentence)  # sentence without mask
    #     # # print('masked tokens list : ',
    #     #     # [pos.item() for pos in printmasked_tokens[0] if pos.item() != 0])  # tokens of words covered by mask

    #     # logits_lm = model(printinput_ids, printsegment_ids, printmasked_pos)
    #     # logits_lm = logits_lm.data.max(2)[1][0].cpu().data.numpy()
    #     # print('predict masked tokens list : ', [pos for pos in logits_lm if pos != 0])  # tokens of predicted noun

    #     # masked_token_list = [pos.item() for pos in printmasked_tokens[0] if pos.item() != 0]
    #     # predicted_mt_list = [pos for pos in logits_lm if pos != 0]

    #     # list_for_output = []
    #     # maskpos = 0
    #     # for w in printinput_ids[0]:
    #     #     if testnumber_dict[w.item()] != '[PAD]':  # if not padded
    #     #         if testnumber_dict[w.item()] == '[MASK]':  # if word got randomly masked
    #     #             list_for_output.append(number_dict[predicted_mt_list[maskpos].item()])
    #     #             maskpos += 1
    #     #         else:  # if word not masked
    #     #             list_for_output.append(testnumber_dict[w.item()])

    #     # newsentence = " ".join(list_for_output)


    # averageLast5 = losses[-5:]
    # average = sum(averageLast5) / 5
    # # print("average last 5 losses:", average)
    # return average


def main():
    run_bert("us")
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()