import torch
import pickle
from data_module.jsonl_data_module import JsonlDataModule
from omegaconf import OmegaConf

config = OmegaConf.load("config/tree_lstm_med_tenth1.yaml")

data_module = JsonlDataModule(config)
data_module.prepare_data()
data_module.setup()

data_dir = "../data"
predict_file = "sgd_global_predictions.pkl"
output_file = "sgd_global_str_preds.pkl"

id2label = data_module._vocabulary.id_to_label

predictions = torch.load(f"{data_dir}/{predict_file}", map_location=torch.device('cpu'))

with open(f"{data_dir}/{output_file}", 'w') as fout:
    for batch in predictions:
        for batch_idx in range(batch.size(1)):
            res = ''
            for idx in batch[:, batch_idx]:
                if idx == 2:
                    continue
                if idx == 3:
                  break
                res += f"{id2label[idx.item()]}|"
            fout.write(res[:-1] + '\n')
