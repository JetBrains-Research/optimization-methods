# Copyright 2021 Dmitry Vilensky-Pasechnyuk

from collections import defaultdict
import numpy as np

from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.chrf_score import corpus_chrf, sentence_chrf
from rouge import Rouge
from chrFpp import computeChrF
from bert_score import score as bert_score

from time import perf_counter


class Metrics:
    def __init__(self, hyps, refs):
        self.hyps = hyps
        self.refs = refs

    def get_statistics(self, with_symbolic=False, bert=False):
        result = {
            'scores': {},
            'score': {}
        }
        for func in dir(Metrics):
            if callable(getattr(Metrics, func)) and func[0] != '_' and func != 'get_statistics':
                if (not with_symbolic and 'symb' in func) or (not bert and func == 'bert'):
                    continue
                print(f'Calculating {func} metric')
                start = perf_counter()
                metric_result = getattr(Metrics, func)(self)
                result['scores'].update(metric_result['scores'])
                result['score'].update(metric_result['score'])
                finish = perf_counter()
                print('Metric is calculated. Elapsed time:', finish - start)

        return result

    def bert(self):
        # TODO: add a proper scaling
        (P, R, F), hashname = bert_score(
            self.hyps, self.refs, lang="en", return_hash=True)
        
        return {
            'scores': {
                'bert-P': P.cpu().detach().numpy(),
                'bert-R': R.cpu().detach().numpy(),
                'bert-F': F.cpu().detach().numpy()
            },
            'score': {
                'bert-P': P.mean().item(),
                'bert-R': R.mean().item(),
                'bert-F': F.mean().item()
            }
        }

    def meteor(self):
        scores = []
        for hyp, ref in zip(self.hyps, self.refs):
            scores.append(single_meteor_score(ref, hyp))

        return {
            'scores': {
                'meteor': np.array(scores)
            },
            'score': {
                'meteor': np.mean(scores)
            }
        }

    def bleu(self):

        # TODO: there are in fact the standard weights of BLEU, isn't it?
        def bleu_weights(l):
            length = len(l)
            if length >= 4:
                return (0.25, 0.25, 0.25, 0.25)
            wt = [1 / length for _ in range(length)]
            wt.extend([0 for _ in range(4-length)])
            return tuple(wt)

        scores = []

        for hyp, ref in zip(self.hyps, self.refs):
            scores.append(sentence_bleu([ref], hyp, weights=bleu_weights(ref)))

        return {
            'scores': {
                'bleu': np.array(scores)
            },
            'score': {
                'bleu': corpus_bleu([[ref] for ref in self.refs], self.hyps)
            }
        }
    
    def chrF(self):
        scores = []

        for hyp, ref in zip(self.hyps, self.refs):
            scores.append(sentence_chrf([ref], hyp))

        return {
            'scores': {
                'chrF': np.array(scores)
            },
            'score': {
                'chrF': corpus_chrf([[ref] for ref in self.refs], self.hyps)
            }
        }

    def chrfpp(self):

        total_f1, _, _, _, sentence_scores = computeChrF(self.refs, self.hyps)

        return {
            'scores': {
                'chrf++': np.array(sentence_scores)
            },
            'score': {
                'chrf++': total_f1
            }

        }
    
    # def chrFplusplus(self):
    #     raise NotImplementedError()

    # def precision(self):
    #     raise NotImplementedError()

    # def recall(self):
    #     raise NotImplementedError()

    # def f1(self):
    #     raise NotImplementedError()

    def symb_rouge_l(self):
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

    def rouge(self):
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
            score_dict[title] = scores_dict[title].mean()

        return {
            'scores': scores_dict,
            'score': score_dict
        }

    def prec_rec_f1(self):
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

            prec_sent, rec_sent, f1_sent = calculate_one_token_metrics(tp, fp, fn)

            prec_scores.append(prec_sent)
            rec_scores.append(rec_sent)
            f1_scores.append(f1_sent)

            true_positive += tp
            false_positive += fp
            false_negative += fn

        prec_total, rec_total, f1_total = calculate_one_token_metrics(true_positive, false_positive, false_negative)

        return {
            'scores': {
                'precision': np.array(prec_scores),
                'recall': np.array(rec_scores),
                'f1': np.array(f1_scores)
            },
            'score': {
                'precision': prec_total,
                'recall': rec_total,
                'f1': f1_total
            }
        }


def calculate_one_token_metrics(true_positive: int, false_positive: int, false_negative: int):
    precision, recall, f1 = 0.0, 0.0, 0.0
    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


if __name__ == "__main__":
    refs = ['мама мыла раму', 'рыба дышит под водой']
    hyps = ['мама мыла раму', 'рыба продается в магазине']

    metrics = Metrics(hyps, refs)
    print(metrics.get_statistics())
