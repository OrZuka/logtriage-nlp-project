"""Evaluation helpers."""
from sklearn.metrics import accuracy_score, f1_score
def field_metrics(true,pred): return {'accuracy':accuracy_score(true,pred),'macro_f1':f1_score(true,pred,average='macro')}
def evidence_metrics(gold,pred):
    tp=fp=fn=0
    for g,p in zip(gold,pred):
        g=set(g); p=set(p); tp+=len(g&p); fp+=len(p-g); fn+=len(g-p)
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return {'precision':precision,'recall':recall,'f1':f1}
