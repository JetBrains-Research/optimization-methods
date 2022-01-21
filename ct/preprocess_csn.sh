# python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml java train
# python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml java valid
# python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml java test

python3.8 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml java train
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml java valid
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml java test

python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml python train
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml python valid
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-csn.yaml python test

python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml python train
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml python valid
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml python test