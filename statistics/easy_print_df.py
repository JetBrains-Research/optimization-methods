#!/usr/bin/env python3

import pandas as pd
pd.set_option('display.max_columns', 500)

print(pd.DataFrame(
    pd.read_csv("ct_javamed0.1_metrics.csv"), 
    columns=["Method", "rouge-1F", "rouge-2F", "rouge-lF", "chrF", "chrFpp", "meteor", "bert-F", "bleu", "f1"]
).sort_values("chrF").T)