import random
import torch
import torch.optim as optim
import torch.nn as nn
import model_architecture
import batches
import format_text

model_architecture
batches
format_text

model = model_architecture.get_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=8e-7)
length_sentences, number_dict = format_text.get_training_vars()
testsentences, testnumber_dict = format_text.get_testing_output()
batch_num = 0


for i in range(0, round(length_sentences / 64)):  # loops through all sentences
    trainingbatch = batches.make_training_batch(i)  # making batch
    input_ids, segment_ids, masked_tokens, masked_pos = map(torch.LongTensor,
                                                            zip(*trainingbatch))
    logits_lm = model(input_ids, segment_ids, masked_pos)
    loss = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
    loss_print = loss / 128  # dividing loss over 12 pairs
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    batch_num += 1
    print(f'Batch Number: {batch_num} | Loss: {loss_print:.4f}')

index = random.randrange(len(testsentences) - 1)
old_sentence = testsentences[index] + testsentences[index + 1]
printinput_ids, printsegment_ids, printmasked_tokens, printmasked_pos = map(torch.LongTensor,
                                                                            zip(*batches.make_testing_sentence(index)))

masked_sentence = " ".join(
    [testnumber_dict[w.item()] for w in printinput_ids[0] if testnumber_dict[w.item()] != '[PAD]'])
# print('masked sentence : ', masked_sentence)  # sentence with mask still in
# print('actual sentence : ', old_sentence)  # sentence without mask
# print('masked tokens list : ',
      # [pos.item() for pos in printmasked_tokens[0] if pos.item() != 0])  # tokens of words covered by mask

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
