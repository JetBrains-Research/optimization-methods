#!/usr/bin/env python3
"""
Copyright 2021 Dmitry Vilensky-Pasechnyuk

This is the "textmetric" module for evaluation of hypothesis-reference text metrics
like BLEU, ROUGE etc. for the QA, MT, code summarization and other practical tasks.

The framework of the module consist of one class ``Metrics`` with a main function ``get_statistics``.
For example, with the latter we can simply evaluate all the standard metrics for our sentences:

>>> hyps = ['мама мыла раму', 'рыба продается в магазине']
>>> refs = ['мама мыла раму', 'рыба дышит под водой']
>>> Metrics(hyps, refs).get_statistics()['score']['rouge-2P']
0.5

If we need to calculate only the some specific metrics, we can specify their list:

>>> Metrics(hyps, refs).get_statistics(stats={'meteor'})['score'].keys()
dict_keys(['meteor'])

"""

from collections import defaultdict
import numpy as np

# from nltk.translate.meteor_score import single_meteor_score
# from sacrerouge.metrics import Meteor
from sacrebleu import corpus_bleu, sentence_bleu
from sacrebleu import corpus_chrf, sentence_chrf
from sacrebleu import corpus_ter, sentence_ter
from rouge import Rouge
from bert_score import score as bert_score

import warnings
from time import perf_counter


class Metrics:
    def __init__(self, hyps: list, refs: list):
        """
        The constructor of ``Metrics`` can raise the error
        only when ``hyps`` size do not match the ``refs`` size:

        >>> Metrics(['one', 'two'], ['only one'])
        Traceback (most recent call last):
            ...
        AttributeError: hyps and refs must be the same length!
        """

        if len(hyps) != len(refs):
            raise AttributeError("hyps and refs must be the same length!")

        self.hyps = hyps
        self.refs = refs

    @staticmethod
    def get_list(stats: set = set(), with_symbolic: bool = False, bert: bool = True):
        metrics_to_evaluate = set(filter(
            lambda func: callable(
                getattr(Metrics, func)) and func[0] != '_' and not func.startswith('get_') and \
                    not ((not with_symbolic and 'symb' in func) or (not bert and func == 'bert')),
            dir(Metrics)
        ))

        unknown_metrics = stats - metrics_to_evaluate
        if len(unknown_metrics) != 0:
            unknown_str_list = ", ".join(unknown_metrics)
            all_str_list = ", ".join(metrics_to_evaluate)
            raise NotImplementedError("\nThe module does not currently support "
                                      "the following of the metrics you requested:\n\033[91m" + unknown_str_list +
                                      "\n\033[0mThese metrics are available:\n\033[92m" + all_str_list)

        if len(stats) != 0:
            return metrics_to_evaluate & stats
        else:
            return metrics_to_evaluate

    def get_statistics(
        self, stats: set = set(), verbose = False,
        with_symbolic: bool = False, bert: bool = True) -> dict:

        if not verbose:
            warnings.filterwarnings("ignore")

        result = {
            'scores': {},
            'score': {}
        }
        final_metrics_list = self.get_list(stats, with_symbolic, bert)

        for func in final_metrics_list:
            if verbose:
                print(f'Calculating {func} metric..')

            start = perf_counter()
            metric_result = getattr(Metrics, func)(self)
            result['scores'].update(metric_result['scores'])
            result['score'].update(metric_result['score'])
            finish = perf_counter()

            if verbose:
                print('Metric is calculated. Elapsed time:', finish - start)
                print()

        return result

    def bert(self) -> dict:
        (P, R, F), hashname = bert_score(
            self.hyps, self.refs, model_type='microsoft/deberta-xlarge-mnli', return_hash=True, rescale_with_baseline=True, lang ="en")

        return {
            'scores': {
                'bert-P': P.cpu().detach().numpy(),
                'bert-R': R.cpu().detach().numpy(),
                'bert-F': F.cpu().detach().numpy()
            },
            'score': {
                'bert-P': round(50 * (1 + P.mean().item()), 1),
                'bert-R': round(50 * (1 + R.mean().item()), 1),
                'bert-F': round(50 * (1 + F.mean().item()), 1)
            }
        }

    def meteor(self) -> dict:
        # technique = Meteor()
        # scores = technique.score_all(self.hyps, [[ref] for ref in self.refs])
        # scores = list(map(lambda x: x['METEOR'], scores))

        return {
            'scores': {
                'meteor': np.zeros(len(self.hyps)) # np.array(scores)
            },
            'score': {
                'meteor': 0.0 # round(100 * np.mean(scores), 1)
            }
        }

    def bleu(self) -> dict:
        """
        Evaluates the BLEU metric.
        """
        scores = []

        for hyp, ref in zip(self.hyps, self.refs):
            scores.append(sentence_bleu(hyp, [ref]).score)

        return {
            'scores': {
                'bleu': np.array(scores)
            },
            'score': {
                'bleu': round(corpus_bleu(self.hyps, [self.refs]).score, 1)
            }
        }
    
    # def ter(self) -> dict:
    #     scores = []

    #     for hyp, ref in zip(self.hyps, self.refs):
    #         scores.append(sentence_ter(hyp, [ref]).score)

    #     return {
    #         'scores': {
    #             'ter': np.array(scores)
    #         },
    #         'score': {
    #             'ter': round(corpus_ter(self.hyps, [self.refs]).score, 1)
    #         }
    #     }

    def chrF(self) -> dict:
        scores = []

        for hyp, ref in zip(self.hyps, self.refs):
            scores.append(sentence_chrf(hyp, [ref]).score)

        return {
            'scores': {
                'chrF': np.array(scores)
            },
            'score': {
                'chrF': round(corpus_chrf(self.hyps, [self.refs]).score, 1)
            }
        }

    def chrFpp(self) -> dict:
        scores = []

        for hyp, ref in zip(self.hyps, self.refs):
            scores.append(sentence_chrf(hyp, [ref], word_order=2).score)

        return {
            'scores': {
                'chrFpp': np.array(scores)
            },
            'score': {
                'chrFpp': round(corpus_chrf(self.hyps, [self.refs], word_order=2).score, 1)
            }
        }

    def symb_rouge_l(self) -> dict:
        rouge = Rouge(metrics=['rouge-l'])
        scores_dict = defaultdict(list)

        hyps_symb = [' '.join(h.replace(' ', '')) for h in self.hyps]
        refs_symb = [' '.join(r.replace(' ', '')) for r in self.refs]

        scores = rouge.get_scores(hyps_symb, refs_symb)
        for score in scores:
            batch = {}
            for sub in score:
                batch.update({sub + k.upper(): v for k,
                             v in score[sub].items()})

            for title in batch:
                scores_dict['symb-' + title].append(batch[title])

        scores_dict = dict(scores_dict)
        score_dict = {}
        for title in scores_dict:
            scores_dict[title] = np.array(scores_dict[title])
            score_dict[title] = scores_dict[title].mean()

        return {
            'scores': scores_dict,
            'score': score_dict
        }

    def rouge(self) -> dict:
        rouge = Rouge()
        scores_dict = defaultdict(list)

        scores = rouge.get_scores(self.hyps, self.refs)
        for score in scores:
            batch = {}
            for sub in score:
                batch.update({sub + k.upper(): v for k,
                             v in score[sub].items()})

            for title in batch:
                scores_dict[title].append(batch[title])

        scores_dict = dict(scores_dict)
        score_dict = {}
        for title in scores_dict:
            scores_dict[title] = np.array(scores_dict[title])
            score_dict[title] = round(scores_dict[title].mean() * 100, 1)

        return {
            'scores': scores_dict,
            'score': score_dict
        }

    def prec_rec_f1(self) -> dict:
        true_positive = false_positive = false_negative = 0
        prec_scores, rec_scores, f1_scores = [], [], []
        for hyp, ref in zip(self.hyps, self.refs):
            tp = fp = fn = 0
            gt_seq = ref.split()
            pred_seq = hyp.split()

            if len(gt_seq) == len(pred_seq) and all([g == p for g, p in zip(gt_seq, pred_seq)]):
                tp += len(gt_seq)
            else:
                for pred_subtoken in pred_seq:
                    if pred_subtoken in gt_seq:
                        tp += 1
                    else:
                        fp += 1
                for gt_subtoken in gt_seq:
                    if gt_subtoken not in pred_seq:
                        fn += 1

            prec_sent, rec_sent, f1_sent = calculate_one_token_metrics(
                tp, fp, fn)

            prec_scores.append(prec_sent)
            rec_scores.append(rec_sent)
            f1_scores.append(f1_sent)

            true_positive += tp
            false_positive += fp
            false_negative += fn

        prec_total, rec_total, f1_total = calculate_one_token_metrics(
            true_positive, false_positive, false_negative)

        return {
            'scores': {
                'precision': np.array(prec_scores),
                'recall': np.array(rec_scores),
                'f1': np.array(f1_scores)
            },
            'score': {
                'precision': round(100 * prec_total, 1),
                'recall': round(100 * rec_total, 1),
                'f1': round(100 * f1_total, 1)
            }
        }


def calculate_one_token_metrics(true_positive: int, false_positive: int, false_negative: int):
    """
    Calculates the precision, recall and f1 scores for presented TP, FP and FN rates:
    
    >>> calculate_one_token_metrics(1, 1, 1)
    (0.5, 0.5, 0.5)

    TP, FP or FN rates can be of ``float`` type, but must be the exact integers:

    >>> calculate_one_token_metrics(1, 1.5, 1)
    Traceback (most recent call last):
        ...
    ValueError: TP, FP and FN rates must be integers!
    """

    if not (float(true_positive).is_integer() and float(false_positive).is_integer() and\
            float(false_negative).is_integer()):
        raise ValueError("TP, FP and FN rates must be integers!")

    precision, recall, f1 = 0.0, 0.0, 0.0
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


if __name__ == "__main__":
    import doctest
    doctest.testmod()
