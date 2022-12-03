import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

NUM_RETURN_SEQUENCES = 3
NUM_BEAMS = 10
MODEL_NAME = "prithivida/parrot_paraphraser_on_T5"
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH_TO_DATASET = '../rxr_train_guide.jsonl'
OUTPUT_PATH = 'rxr_train_new.jsonl'
print(torch_device)

def create_model_and_tokenizer(): 
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def get_responses(input_text, model, tokenizer, max_length):
    print('in get_responses')
    encoding = tokenizer.encode("paraphrase: " + input_text, max_length = max_length, return_tensors='pt')
    print('shape of encoding: ', encoding.shape)
    input_ids = encoding['input_ids'].to(torch_device)
    print('shape of input_ids: ', input_ids.shape)
    return model.generate(**batch,
                         max_length=max_length,
                         num_beams = 5,
                         num_return_sentences = NUM_RETURN_SENTENCES)

def load_dataset(path_to_data):
    new_data = []
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

def write_and_generate_parallel_for_single_annotation(annotation, model, tokenizer, f, pad_to_len):
    """
    Parameters
    ----------
    annotation : Dictionary
                 Represents a single annotation
    model      :
    tokenizer  :
    f          : pointer to open file
    pad_to_len : length to pad sequences to

    Returns an Array[Dictionary] of the synthetically generated annotations. These annotations
    should be identical to the original annotation besides the instruction field, which should
    include the synthetically generated data.
    """
    instruction = annotation['instruction']
    new_instructions = get_responses(instruction, model, tokenizer, pad_to_len)
    print()
    print('old instruction: ')
    print(instruction)
    print()
    print('lenth of responses goten: ', len(new_instructions))

    for new_instruction in new_instructions:
        print('writing w the following new instruction:')
        print(new_instruction)
        annotation['instruction'] = new_instruction
        # json.dump(annotation, f)

    annotation['instruction'] = instruction
    print()

    return new_instructions

def generate_parallel(data, write_to_path):
    """
    Parameters
    ----------
    data          : Array[Dictionary]
    write_to_path : String
    """
    model, tokenizer = create_model_and_tokenizer()
    f = open(write_to_path, 'a')
    # for annotation in data:
    #     new_annotations = write_and_generate_parallel_for_single_annotation(annotation, model, tokenizer, f)

    pad_to_len = max([len(annotation['instruction']) for annotation in data])
    print(pad_to_len)
    print(pad_to_len)

    for i in range(3):
        new_annotations = write_and_generate_parallel_for_single_annotation(data[i], model, tokenizer, f, pad_to_len)
    
    f.close()
    print('done writing and generating!')


if __name__ == '__main__':
    new_data = load_dataset(PATH_TO_DATASET)
    filtered = filter_for_language(new_data, 'en-US')
    generate_parallel(filtered, OUTPUT_PATH)
