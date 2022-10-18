from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'a', 'test','ojbk','ojbk']]
candidate = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, candidate)
print(score)