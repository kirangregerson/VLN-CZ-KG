# VLN-CZ-KG
***Vision-Language-Navigation Agent Research***
This repository generates semantically-paired instructions using a paraphrase model for a VLN task. 

## Setup

It's recommended to use Conda to manage packages. Create a separate conda environment

1. Install dependencies for the data generation.
    1. Create a conda environment
       ```bash
       conda create -n pegasus-paraphrase python=3.9
       ```
    3. Use your preferred method and CUDA version (if applicable) to install PyTorch version 1.10.1, following the instructions from [here](https://pytorch.org/get-started/previous-versions/). ]

2. Create the conda environment for the agent & set up the dependencies
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

