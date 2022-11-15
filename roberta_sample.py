from transformers import pipeline
unmasker = pipeline('fill-mask', model='roberta-base')
print(unmasker("The <mask> is mightier than the sword."))
