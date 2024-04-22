
import BERT_run 

BERT_run
paperLosses = []
for i in range(0,5):
    paperLosses.append(BERT_run.runBert("paper"))


paperAverage = sum(paperLosses) / len(paperLosses)

oursLosses = []
for i in range(0,5):
    oursLosses.append(BERT_run.runBert("ours"))

oursLosses = sum(oursLosses) / len(oursLosses)

print("paper:", paperAverage)
print("us:", oursLosses)