import numpy as np
from sklearn.metrics import confusion_matrix

#same class of labels, differently named in different datasets. Make sure to follow the same set of labels for both the source and the target
'''
01 - secondary incision knife - paracenthesis blade
02 - bonn forceps - 0.12 forceps
03 - charleux cannula - AC Cannula
04 - primary incision knife - keratome
05 - capsulorhexis forceps - utrata forceps
07 - capsulorhexis cystotome - cystotome
08 - hydrodissection cannula - hydrodissection cannula
09 - phacoemulsifier handpiece - phaco
11 - implant injector - IOL injector
13 - suture needle - suture
14 - cotton - eweckell sponge
16 - needle holder - needle driver
22 - irrigation - irrigation
25 - micromanipulator - chopper
'''

def visda_acc(predict, all_label):
    matrix = confusion_matrix(all_label, predict)
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean()
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)
    return aacc, acc

def micro_accuracy(predict, all_label):
    predict_np = np.array(predict)
    all_label_np = np.array(all_label)
    return (predict_np==all_label_np).mean()

