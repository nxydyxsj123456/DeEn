import  numpy as np
from sklearn import  metrics

scores_good=np.load('scores_good.npy')
scores_bad=np.load('scores_bad.npy')



label1=np.zeros(len(scores_good))
label2=np.ones(len(scores_bad))

label=np.append(label1,label2)
pred=np.append(scores_good,scores_bad)

fpr,tpr,thresholds = metrics.roc_curve(label,pred,pos_label=0)

auc= metrics.auc(fpr,tpr)

print(auc)
