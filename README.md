# code2seq experiments
Numerical experiments for the model [code2seq](https://github.com/JetBrains-Research/code2seq)
### important directories
* code2seq/optimizer
  
  New implemented optimization methods
  
### Protocol
* Download dataset using /scripts
* Place dataset in /code2seq/permanent-data
* Run process_dataset.py, enter local path to dataset directory
* Set train/val_holdout to train.# and val.# correspondingly (certain number instead of #)
* Run /code2seq/preprocessing/build_vocabulary.py
* Run experiments!

### third party links
* [W&B plots java-small](https://wandb.ai/dmivilensky/code2seq-java-small)
* [W&B plots java-med (random 10%)](https://wandb.ai/dmivilensky/code2seq-java-med)
