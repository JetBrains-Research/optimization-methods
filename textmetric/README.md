# `textmetric` module
This module allows us to calculate a bunch of text similarity metrics in a centralized and automatic way, aggregating the per sentence and corpus evaluated chracteristics in one resulting dictionary

There is the list of metrics we use:

* BERT (for `en` language)
* meteor
* BLEU (with specific weights)
* chrF / chrF++ (using Maja Popovic impl.)
* rouge / symbolic rouge