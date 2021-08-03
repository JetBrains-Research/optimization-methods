wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
cd ./dataset
rm ../dataset.zip

wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip
rm python.zip

wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
unzip java.zip
rm java.zip

# !wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
# !unzip ruby.zip
# !rm ruby.zip

# !wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
# !unzip javascript.zip
# !rm javascript.zip

# !wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
# !unzip go.zip
# !rm go.zip

# !wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip
# !unzip php.zip
# !rm *.zip

rm *.pkl
python3 preprocess.py
rm -r */final
cd ..

pip3 install tokenizers transformers
pip3 install torch_optimizer

wget https://huggingface.co/huggingface/CodeBERTa-small-v1/raw/main/merges.txt
wget https://huggingface.co/huggingface/CodeBERTa-small-v1/raw/main/vocab.json

pip3 install wandb