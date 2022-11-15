from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text = "Go forward until you reach the wine cabinet."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(model)
