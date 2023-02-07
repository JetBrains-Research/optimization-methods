# Judging Adam: Studying the Performance of Optimization Methods on ML4SE Tasks

This is an official repository containing the experiments reported in
> Pasechnyuk D.A., Prazdnichnykh A., Evtikhiev M., Bryksin T. "Judging Adam: Studying the Performance of Optimization Methods on ML4SE Tasks",

which is submitted to **ICSE NIER'2023**.

Source code for `Code2Seq`, `TreeLSTM` and `CodeTransformer` models, which were used in course of our research, was adapted by us based on existing implementations.
The list of corresponding original repositories follows:
* [Code2Seq](https://github.com/JetBrains-Research/code2seq) by [SpirinEgor](https://github.com/SpirinEgor)
* [TreeLSTM](https://github.com/JetBrains-Research/embeddings-for-trees) by [SpirinEgor](https://github.com/SpirinEgor)
* [CodeTransformer](https://github.com/danielzuegner/code-transformer) by [danielzuegner](https://github.com/danielzuegner)

The `CodeGNN` model's implementation with PyTorch is original, it's based on the Tensorflow implementation from
* [CodeGNN](https://github.com/acleclair/ICPC2020_GNN) by [acleclair](https://github.com/acleclair)

For running training and validation, follow the instructions in corresponding `README.md` files in [code2seq](code2seq), [treelstm](treelstm), [codetransformer](codetransformer) and [codegnn](codegnn).
