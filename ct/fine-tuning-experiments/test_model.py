from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF

from code_transformer.modeling.constants import UNKNOWN_TOKEN, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, \
    NUM_SUB_TOKENS_METHOD_NAME
from code_transformer.modeling.modelmanager.code_transformer import CodeTransformerModelManager
from code_transformer.preprocessing.datamanager.base import batch_to_device, batch_filter_distances
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDatasetNoPunctuation
from code_transformer.preprocessing.graph.binning import ExponentialBinning, EqualBinning
from code_transformer.preprocessing.graph.distances import DistanceBinning
from code_transformer.preprocessing.graph.transform import TokenDistancesTransform
from code_transformer.utils.metrics import f1_score
from code_transformer.env import DATA_PATH_STAGE_2

BATCH_SIZE = 8
LIMIT_TOKENS = 1000  # MAX_NUM_TOKENS


def calc_chrf(prediction: str, target: str) -> float:
    chrf_metric = CHRF()
    result = chrf_metric.sentence_score(prediction, [target])
    return result.score


def calc_bleu(prediction: str, target: str) -> float:
    bleu_metric = BLEU(effective_order=True, smooth_method="add-k")
    result = bleu_metric.sentence_score(prediction, [target])
    return result.score


def calculate_metrics(run_id: str, snapshot_iteration: str, partition: str = "test"):
    model_manager = CodeTransformerModelManager()

    model = model_manager.load_model(run_id, snapshot_iteration, gpu=True)
    model = model.eval()
    model = model.cuda()

    config = model_manager.load_config(run_id)
    data_manager = CTBufferedDataManager(DATA_PATH_STAGE_2,
                                         config['data_setup']['language'],
                                         partition=partition,
                                         shuffle=False)
    vocabularies = data_manager.load_vocabularies()
    if len(vocabularies) == 3:
        word_vocab, _, _ = vocabularies
    else:
        word_vocab, _, _, _ = vocabularies

    token_distances = None
    if TokenDistancesTransform.name in config['data_transforms']['relative_distances']:
        num_bins = data_manager.load_config()['binning']['num_bins']
        distance_binning_config = config['data_transforms']['distance_binning']
        if distance_binning_config['type'] == 'exponential':
            trans_func = ExponentialBinning(distance_binning_config['growth_factor'])
        else:
            trans_func = EqualBinning()
        token_distances = TokenDistancesTransform(
            DistanceBinning(num_bins, distance_binning_config['n_fixed_bins'], trans_func))

    use_pointer_network = config['data_setup']['use_pointer_network']

    dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager,
                                                      num_sub_tokens_output=NUM_SUB_TOKENS_METHOD_NAME,
                                                      use_pointer_network=use_pointer_network,
                                                      max_num_tokens=LIMIT_TOKENS,
                                                      token_distances=token_distances)

    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=BATCH_SIZE)

    relative_distances = config['data_transforms']['relative_distances']

    pad_id = word_vocab[PAD_TOKEN]
    unk_id = word_vocab[UNKNOWN_TOKEN]

    f1_scores = []
    precisions = []
    recalls = []

    predictions = []
    labels = []
    progress = tqdm(enumerate(dataloader), total=int(data_manager.approximate_total_samples() / BATCH_SIZE))
    for i, batch in progress:
        batch = batch_filter_distances(batch, relative_distances)
        batch = batch_to_device(batch)

        label = batch.labels.detach().cpu()
        with torch.no_grad():
            output = model.forward_batch(batch).cpu()

        f1, prec, rec = f1_score(output.logits, label, pad_id=pad_id, unk_id=unk_id,
                                 output_precision_recall=True)
        f1_scores.append(f1)
        precisions.append(prec)
        recalls.append(rec)

        batch_logits = output.logits.detach().cpu()

        predictions.extend(batch_logits.argmax(-1).squeeze(1))
        labels.extend(label.squeeze(1))

        progress.set_description()
        del batch

    data_manager.shutdown()

    predictions = torch.stack(predictions)
    labels = torch.stack(labels)

    ignore_index = [word_vocab.vocabulary[token] for token in [UNKNOWN_TOKEN, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN]]
    to_words = lambda tensor: " ".join(
        word_vocab.reverse_vocabulary[token.item()] for token in tensor if
        token.item() not in ignore_index and token.item() in word_vocab.reverse_vocabulary.keys())

    predictions = [to_words(prediction) for prediction in predictions]
    targets = [to_words(target) for target in labels]

    bleus = []
    chrfs = []
    for p, t in zip(predictions, targets):
        if p == "" or t == "":
            bleus.append(0)
            chrfs.append(0)
            continue
        bleus.append(calc_bleu(p, t))
        chrfs.append(calc_chrf(p, t))

    metric_names = ["f1", "precision", "recall", "bleu", "chrf"]
    metric_data = [f1_scores, precisions, recalls, bleus, chrfs]

    results = {metric: np.asarray(data).mean() for metric, data in zip(metric_names, metric_data)}

    return results


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("run_id", type=str)
    parser.add_argument("snapshot_iteration", type=str)
    parser.add_argument("partition", type=str, choices=['train', 'valid', 'test'], default='test')
    args = parser.parse_args()

    print(calculate_metrics(args.run_id, args.snapshot_iteration, args.partition))
