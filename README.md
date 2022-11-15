# VLN-CZ-KG
***Vision-Language-Navigation Agent Research***

Environment Setup
 - Download Miniconda, create environment
 - After activating, run:
  - conda install transformers
    - This module contains a number of default transformers, including roBERTa and Google ViT
  - conda install Pillow
    - This is a digital image processing module
 - All present python scripts should now be functional.

Scripts
 - roberta_sample.py
  - Attempts to fill in <mask> tokens within a sentence with the most likely word
 - vit_sample.py
  - Guesses the contents of the linked image (which can be changed).
