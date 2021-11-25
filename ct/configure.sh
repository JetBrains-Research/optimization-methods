pip3 install -r requirements.txt

DIR=$(pwd);
FILE_ENV=${HOME}"/.config/code_transformer/.env"
echo "export WD="$DIR > $FILE_ENV
echo "export CODE_TRANSFORMER_DATA_PATH=\${WD}/data" >> $FILE_ENV
echo "export CODE_TRANSFORMER_BINARY_PATH=\${WD}/bin" >> $FILE_ENV
echo "export CODE_TRANSFORMER_MODELS_PATH=\${WD}/models" >> $FILE_ENV
echo "export CODE_TRANSFORMER_LOGS_PATH=\${WD}/logs" >> $FILE_ENV

mkdir data