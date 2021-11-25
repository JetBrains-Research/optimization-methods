# cd data
# wget https://s3.amazonaws.com/code2seq/datasets/java-small.tar.gz
# tar -xf java-small.tar.gz
# rm java-small.tar.gz
# mv java-small raw/
# cd raw
# mkdir code2seq
# mv java-small code2seq/
# cd ../..

python3 -m scripts.extract-java-methods java-small-vocab

python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-code2seq.yaml java-small-vocab train
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-code2seq.yaml java-small-vocab valid
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-1-code2seq.yaml java-small-vocab test

python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml java-small-vocab train
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml java-small-vocab valid
python3 -m scripts.run-preprocessing code_transformer/experiments/preprocessing/preprocess-2-small-vocab.yaml java-small-vocab test
