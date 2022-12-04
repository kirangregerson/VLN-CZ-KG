import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import json

NUM_RETURN_SEQUENCES = 3
NUM_BEAMS = 10
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch_device)

def create_model_and_tokenizer(): 
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return model, tokenizer

def get_response(input_text, model, tokenizer, max_length):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=max_length, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=max_length,num_beams=NUM_BEAMS, num_return_sequences=NUM_RETURN_SEQUENCES, temperature=2)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

def load_dataset(path_to_data):
    new_data = []
    # lines = open(path_to_data).read().splitlines()
    for line in open(path_to_data):
        new_data.append(json.loads(line))
    return new_data

def filter_for_language(data, language):
    """
    Filters the provided list of annotations for only annotations containing instructions in the 
    provided language.

    Parameters
    ----------
    data     : Array[Dictionary]
               JSON data read in as a list of dictionaries
    language : String

    Returns an Array[Dictionary] of data that's been filtered for the provided language
    """
    return list(filter(lambda annotation: annotation['language'] == language, data))

def write_and_generate_parallel_for_single_annotation(annotation, model, tokenizer, f, max_length):
    """
    Parameters
    ----------
    annotation : Dictionary
                 Represents a single annotation
    model      :
    tokenizer  :

    Returns an Array[Dictionary] of the synthetically generated annotations. These annotations
    should be identical to the original annotation besides the instruction field, which should
    include the synthetically generated data.
    """
    instruction = annotation['instruction']
    print('old instruction: ')
    print(instruction)
    new_instructions = get_response(instruction, model, tokenizer, max_length)
    print()

    for new_instruction in new_instructions:
        annotation['instruction'] = new_instruction
        print('new instruction')
        print(new_instruction)
#        json.dump(annotation, f)
    print('-' * 100)

    annotation['instruction'] = instruction

    return new_instructions

def generate_parallel(data, write_to_path):
    model, tokenizer = create_model_and_tokenizer()
    pad_to_len = max([len(annotation['instruction']) for annotation in data])
    f = open(write_to_path, 'a')
    #for annotation in data:
    #    new_annotations = write_and_generate_parallel_for_single_annotation(annotation, model, tokenizer, f, pad_to_len)
    for i in range(3):
         new_annotations = write_and_generate_parallel_for_single_annotation(data[i], model, tokenizer, f, pad_to_len)
    f.close()
    print('done writing and generating!')


if __name__ == '__main__':
    new_data = load_dataset('../rxr_train_guide.jsonl')
    filtered = filter_for_language(new_data, 'en-US')
    generate_parallel(filtered, 'rxr_train_new')
