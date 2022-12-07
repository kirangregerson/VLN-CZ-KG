import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import json
import copy

NUM_RETURN_SEQUENCES = 3
NUM_BEAMS = 10
LOAD_DATA_PATH = '../../rxr_data/RxR_VLNCE_v0/train/train_guide.json'
LANG = 'en-US'
OUT_PATH = 'rxr_train_new'

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cpu'

print("training on: ", torch_device)

def create_model_and_tokenizer(): 
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return model, tokenizer

def get_response(input_text, model, tokenizer, max_length):
    batch = tokenizer([input_text],truncation=True,padding='longest',max_length=max_length, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=max_length,num_beams=NUM_BEAMS, num_return_sequences=NUM_RETURN_SEQUENCES, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def load_dataset(path_to_data):
    """
    Parameters
    ----------
    path_to_data : String
                   path to .json file
    
    Returns
    -------
    Array of episode dictionary objects.
    """
    f = open(path_to_data)

    episodes = json.load(f)
    return episodes['episodes']

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
    return list(filter(lambda annotation: annotation['instruction']['language'] == language, data))

def generate_parallel(data, write_to_path):
    model, tokenizer = create_model_and_tokenizer()
    f = open(write_to_path, 'a')
    episodes = {"episodes": []}

    for annotation in data:
        new_annotations = write_and_generate_parallel_for_single_annotation(annotation, model, tokenizer, f)
        for new_anno in new_annotations:
            episodes["episodes"].append(new_anno)

    json.dump(episodes, f)
    f.close()
    print('done writing and generating!')

def write_and_generate_parallel_for_single_annotation(annotation, model, tokenizer, f):
    """
    Parameters
    ----------
    annotation : Dictionary
                 Represents a single annotation
    model      :
    tokenizer  :
    f          : open file pointer

    Returns an Array[Dictionary] of the synthetically generated annotations. These annotations
    should be identical to the original annotation besides the instruction field, which should
    include the synthetically generated data.
    """
    instruction = annotation['instruction']['instruction_text']
    new_instructions = generate_parallel_for_instruction(instruction, model, tokenizer)

    new_dicts = []
    new_dicts.append(annotation)

    for new_instruction in new_instructions:
        new_anno = copy.deepcopy(annotation)
        new_anno['instruction']['instruction_text'] = new_instruction
        new_dicts.append(new_anno)

    return new_dicts

def generate_parallel_for_instruction(instruction, model, tokenizer):
    """
    Parameters
    ----------
    instruction : String
                  unprocessed instruction from the jsonl

    Returns
    -------
    Returns a list of strings
    """
    instr_segs, max_length = process_instruction_into_segments(instruction)

    new_instrs = [] # 2d list of strings

    for seg in instr_segs:
        if len(seg.strip()) > 0:
            new_instrs.append(get_response(seg, model, tokenizer, max_length))


    new_instrs_joined = [] # list of strings
    for seq in range(NUM_RETURN_SEQUENCES):
        new_instr = []
        for line in new_instrs:
            new_instr.append(line[seq])
        new_instrs_joined.append(' '.join(new_instr))

    return new_instrs_joined

def process_instruction_into_segments(instruction):
    instr_sents = instruction.split('.')
    max_len = 0

    segments = []
    for sent in instr_sents:
        if len(sent.split()) > 20:
            # if sentence is too long, split by commas and append each comma as separate
            # segment
            for seg in sent.split(','):
                segments.append(seg)
                if len(seg) > max_len:
                    max_len = len(seg)
        else:
            segments.append(sent)
            if len(sent) > max_len:
                max_len = len(sent)
    return segments, max_len + 1

if __name__ == '__main__':
    new_data = load_dataset(LOAD_DATA_PATH)
    filtered = filter_for_language(new_data, LANG)
    generate_parallel(filtered, OUT_PATH)
