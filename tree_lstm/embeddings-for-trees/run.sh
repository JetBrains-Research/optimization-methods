#!/usr/bin/env bash

python train_treelstm.py -cn tree_lstm_codexglue_docstrings_Lamb
python train_treelstm.py -cn tree_lstm_codexglue_docstrings_LaLamb

