# VLN-CZ-KG
***Vision-Language-Navigation Agent Research***

This repository generates semantically-paired instructions for Room-to-Room. Generating semantically-paired instructions is treated as a paraphrasing task. Paraphrased sequences are generated using a [finetuned Pegasus model](https://huggingface.co/tuner007/pegasus_paraphrase). Results were tested and compared on the [Discrete-Continuous-VLN model](https://github.com/YicongHong/Discrete-Continuous-VLN).

## Setup

It's recommended to use Conda to manage packages. Create separate conda environments, one to manage the data generation, and another to run the agent. 

### Install dependencies for the data generation.
1. Create a conda environment
   ```bash
   conda create -n pegasus-paraphrase python=3.9
   ```
2. Use your preferred method and CUDA version (if applicable) to install PyTorch version 1.10.1, following the instructions from [here](https://pytorch.org/get-started/previous-versions/). 
3. After cloning this repo, run the following:
   ```bash
   cd VLN-CZ-KG/data_generation
   pip install -r requirements.txt
   ```

### Install dependencies for the agent.
1. After cloning this repo, initialize the submodules with the following:
```bash
cd VLN-CZ-KG
git submodule update --init --recursive
git submodule foreach git pull origin main

cd data_generation
pip install -r requirements.txt
```

2. Run the following:
```bash
cd Discrete-Continuous-VLN
```
and follow the instructions in their `README.md` to install dependencies, including their instructions to install habitat-sim and habitat-lab. It is not necessary to download the connectivity graphs, but follow their instructions for downloading the Matterport3d scene data.

## Run
### Data generation
1. Make sure you're in the correct Conda environment: `conda activate pegasus-paraphrase`
2. Check that the constants at the top of `data_generation/synthetic_data_generation_r2r.py` point to the correct file paths. (tamper with the other constants at your own risk.)
    1. The `LOAD_DIR` string should contain the path to datasets folder that will be modified/paraphrased. In other words, the `LOAD_DIR` path should contain the original dataset. The `LOAD_DIR` folder should have the following structure:
    ```
    LOAD_DIR/
    |--- test/
    |    |--- test.json.gz
    |
    |--- train/
    |    |--- train.json.gz
    |    |--- train_gt.json.gz 
    |
    |--- val_seen/
    |    |--- val_seen.json.gz
    |    |--- val_seen_gt.json.gz 
    |
    |--- val_unseen/
    |    |--- val_unseen.json.gz
    |    |--- val_unseen_gt.json.gz 
    ```
    2. The `OUT_DIR` string should contain the path to the output datasets folder. Make sure that the specified folder already has the following structure because the script does not create new directories:
    ```
    OUT_DIR/
    |--- test/
    |
    |--- train/
    |
    |--- val_seen/
    |
    |--- val_unseen/
    ```  
3. Run the following: `python synthetic_data_generation_r2r.py`
    1. The script took approximately 2-3 hours to run on a GTX3080 GPU.
4. Navigate to your `OUT_DIR` path.
    1. For any newly generated .json files, run `gzip [generated_file_name].json` 
    2. For any other files that existed in the original `LOAD_DIR`, copy them over to the corresponding place in `OUT_DIR`. 

After all that work, you can run the agent :)

## Agent
1. Ensure that you're in the correct Conda environment: `conda activate dcvln`
2. Navigate to the Discrete-Continuous-VLN directory `cd Discrete-Continuous-VLN`
3. Edit `run_CMA.bash` to uncomment the training task. Edit the `exp_name` flag to edit what folder checkpoints are saved to. 
4. Edit `habitat_extensions/config/vlnce_task.yaml`. Modify any `data/datasets...` paths to point to the `OUT_DIR` specified in the previous step. These paths should be relative to the Discrete-Continuous-VLN directory.
    1. For example, if `OUT_DIR = "Discrete-Continuous-VLN/data/datasets/paraphrased"`, then the line `GT_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz` would be modified to be: `GT_PATH: data/datasets/paraphrased/{split}/{split}_gt.json.gz`
6. Refer to instructions [here](https://github.com/YicongHong/Discrete-Continuous-VLN/tree/f3368414b1509b532c24717cc813f361ababce8e#running) to train and run.
7. After training, refer to Discrete-Continuous-VLN instructions to evaluate. Make sure to modify the eval flag in `run_CMA.bash` to point to the correct checkpoint from the latest training.
