
import BERT_run 

BERT_run
paperLosses = []
BERT_run.run_bert("paper")
# for i in range(0,20):
#     paperLosses.append(BERT_run.run_bert("paper"))


# paperAverage = sum(paperLosses) / len(paperLosses)

# oursLosses = []
# for i in range(0,20):
#     oursLosses.append(BERT_run.run_bert("SSL"))

# oursLosses = sum(oursLosses) / len(oursLosses)

# print("paper:", paperAverage)
# print("us:", oursLosses)