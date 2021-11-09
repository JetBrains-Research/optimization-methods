# cd ~/code-transformer
# rm -rf data/stage1/python
# rm -rf data/stage2/python

python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml python train
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml python valid
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml python test

python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2.yaml python train
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2.yaml python valid
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2.yaml python test