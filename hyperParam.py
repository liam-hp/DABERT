
import BERT_run 

BERT_run
losses = []
for i in range(0,5):
    losses.append(BERT_run.runBert())


print(losses)