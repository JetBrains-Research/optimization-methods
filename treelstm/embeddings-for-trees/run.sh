#!/usr/bin/env bash

#python train_treelstm.py -cn tree_lstm_codexglue_docstrings_Lamb
#python train_treelstm.py -cn tree_lstm_codexglue_docstrings_LaLamb

#python train_treelstm.py -cn tree_lstm_javaxglue_Adamax
#python train_treelstm.py -cn tree_lstm_javaxglue_Yogi
#python train_treelstm.py -cn tree_lstm_javaxglue_DiffGrad
#python train_treelstm.py -cn tree_lstm_javaxglue_LaAdamax
#python train_treelstm.py -cn tree_lstm_javaxglue_LaYogi
#python train_treelstm.py -cn tree_lstm_javaxglue_LaDiffGrad
#python train_treelstm.py -cn tree_lstm_javaxglue_Adamax_warmup
#python train_treelstm.py -cn tree_lstm_javaxglue_Yogi_warmup
#python train_treelstm.py -cn tree_lstm_javaxglue_DiffGrad_warmup
#python train_treelstm.py -cn tree_lstm_javaxglue_Yogi_warmup
#python train_treelstm.py -cn tree_lstm_javaxglue_LaYogi_warmup
#python train_treelstm.py -cn tree_lstm_javaxglue_LaDiffGrad_warmup
#python train_treelstm.py -cn tree_lstm_javaxglue_SGD_warmup
#python train_treelstm.py -cn tree_lstm_javaxglue_LaSGD_warmup

python train_treelstm.py -cn tree_lstm_javaxglue_NovoGrad_warmup
python train_treelstm.py -cn tree_lstm_javaxglue_Apollo_warmup
python train_treelstm.py -cn tree_lstm_javaxglue_Adadelta_warmup
python train_treelstm.py -cn tree_lstm_javaxglue_A2GradExp_warmup
python train_treelstm.py -cn tree_lstm_javaxglue_AdaBound_warmup
python train_treelstm.py -cn tree_lstm_javaxglue_Nadam_warmup




