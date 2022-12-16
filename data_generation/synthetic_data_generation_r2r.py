import torch
import gzip
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import json
import copy
import traceback

NUM_RETURN_SEQUENCES = 2
NUM_BEAMS = 10
LOAD_DIR = '/home/grail/Discrete-Continuous-VLN/data/datasets/R2R_VLNCE_v1-2_preprocessed/'
OUT_DIR = '/home/grail/Discrete-Continuous-VLN/data/datasets/R2R_VLNCE_paraphrased/'
SPLITS = ['train', 'val_seen']
SENT_SPLIT_LEN = 10 # in words

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("training on: ", torch_device)

def create_model_and_tokenizer(): 
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return model, tokenizer

def get_response(input_text, model, tokenizer, max_length):
    batch = tokenizer([input_text],truncation=True,padding='max_length',max_length=SENT_SPLIT_LEN, return_tensors="pt").to(torch_device)
    # !!! Some parameter here is causing an embedding index-out-of-range error.
    # len(tokenizer) == model.config.vocab_size == 96013
    # Is an index
    try:
        translated = model.generate(**batch,max_length=SENT_SPLIT_LEN,num_beams=NUM_BEAMS, num_return_sequences=NUM_RETURN_SEQUENCES, temperature=1.5)
    except Exception:
        print("errored out on ", input_text)
        traceback.print_exc()
        print('max_length', max_length)
        print('-'*100)
        # print(batch)
        print('-'*100)
        print('length of input', len(input_text))
        raise Exception("stupid index out of range of self error")
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def load_dataset(path_to_data):
    """
    Parameters
    ----------
    path_to_data : String
                   path to .json.gz file
    
    Returns
    -------
    Array of episode dictionary objects.
    """
    with gzip.open(path_to_data, "rt") as f:
        data = json.load(f)
    return data

def generate_parallel(data, write_to_path):
    model, tokenizer = create_model_and_tokenizer()
    f = open(write_to_path, 'w')
    f_temp = open('generated_sentences.txt', 'w')
    episodes = {"episodes": []}
    annotations = data["episodes"]
    episodes["instruction_vocab"] = data["instruction_vocab"]
    num_annotations = len(annotations)
    i = 0

    for annotation in annotations:
        print('on annotation ', i, ' out of ', num_annotations)
        print()
        new_annotations = write_and_generate_parallel_for_single_annotation(annotation, model, tokenizer, f)
        for new_anno in new_annotations:
            f_temp.write(str(new_anno))
            episodes["episodes"].append(new_anno)
        i += 1

    json.dump(episodes, f)
    f.close()
    f_temp.close()
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
    try:
        new_instructions = generate_parallel_for_instruction(instruction, model, tokenizer)
    except Exception as e:
        if str(e) == "stupid index out of range of self error":
            new_instructions = []
        elif "RuntimeError: CUDA error: device-side assert triggered" in str(e):
            new_instructions = []
        else:
            raise

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
    #print('trying to see if grand is the bad word')
    #get_response('grand grand grand grand grand grand grand', model, tokenizer, max_length)

    new_instrs = [] # 2d list of strings composed of the paraphrased instructions from the model

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
    	# Array of words in the sentence
        if len(sent.split()) > SENT_SPLIT_LEN:
            # if sentence is too long, split by commas and append each comma as separate
            # segment
            for seg in sent.split(','):
            	#segments.append(seg)
            	#if len(seg) > max_len:
            #		max_len = len(seg)
            	# If clause is too long, do:
            	# Option 1: Leave unedited, copy back into each synthetic instruction at
            	#	the appropriate index
            	# --> Option 2: Arbitrarily split sentence every [20] words
            	# Option 3: Don't replicate this instruction
            	# Error is likely being caused by a 'seg' being longer than 20 (but still
            	#	getting added).
                if len(seg.split()) > SENT_SPLIT_LEN:
            	    ceil = (len(seg.split()) // SENT_SPLIT_LEN) + 1
            	    i = 0
            	    for i in range(ceil):
                        if i == ceil - 1:
                            seg_fragment = seg.split()[i * SENT_SPLIT_LEN:]
                            seg_frag_joined = ' '.join(seg_fragment)
                        else:
                            seg_fragment = seg.split()[i * SENT_SPLIT_LEN:(i+1) * SENT_SPLIT_LEN]       
                            seg_frag_joined = ' '.join(seg_fragment)
                        segments.append(seg_frag_joined)	    
                        if len(seg_frag_joined) > max_len:
                            max_len = len(seg_frag_joined)
                else:
            	    segments.append(seg)
            	    if len(seg) > max_len:
                        max_len = len(seg)
        else:
            segments.append(sent)
            if len(sent) > max_len:
                max_len = len(sent)
    return segments, max_len + 1

def process_for_in_path_and_out_path(in_path, out_path):
    new_data = load_dataset(in_path)
    generate_parallel(new_data, out_path)

def process_for_all_splits():
    for split in SPLITS:
        in_path = LOAD_DIR + split + "/" + split + ".json.gz"
        out_path = OUT_DIR + split + "/" + split + ".json"
        process_for_in_path_and_out_path(in_path, out_path)


if __name__ == '__main__':
    process_for_all_splits()
