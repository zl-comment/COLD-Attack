import torch
import numpy as np
import time
import wandb
from distinct_n import distinct_n_sentence_level, distinct_n_corpus_level

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

from evaluation.bert_score.bert_score import BertScore
from evaluation.bleu.bleu import Bleu
from evaluation.cider.cider import Cider
from evaluation.rouge.rouge import Rouge
from evaluation.meteor.meteor import Meteor
from collections import defaultdict
ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
class Scorer:
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (BertScore(), "Bert Score")
        ]

    def compute_dns(self):
        # 将所有句子合并为一个列表，计算语料库级别的 Distinct-1 和 Distinct-2 分数
        all_sentences = []
        for key, gt_sentences in self.gt.items():
            all_sentences.extend(gt_sentences)  # 合并所有句子

        # 计算整个语料库的 Distinct-1 和 Distinct-2 分数
        distinct_corpus_1 = distinct_n_corpus_level(all_sentences, 1)
        distinct_corpus_2 = distinct_n_corpus_level(all_sentences, 2)

        # 返回计算的 DNS 分数
        return {"DNS_1": distinct_corpus_1, "DNS_2": distinct_corpus_2}

    def compute_scores(self):
        # Compute DNS directly using compute_dns method
        dns_scores = self.compute_dns()

        # Create a dictionary for storing all scores
        total_scores = {}

        # First handle DNS separately
        for key, value in dns_scores.items():
            total_scores[f"DNS_{key}"] = value

        # Now handle other scorers
        for scorer, method in self.scorers:
            try:
                score, scores = scorer.compute_score(self.gt, self.ref)
                if type(method) == list:
                    # Store each BLEU score individually
                    for sc, m in zip(score, method):
                        total_scores[m] = sc
                else:
                    total_scores[method] = score
            except Exception as e:
                print(f"Error computing score for {method}: {e}")
                continue

        # 打印分数
        for key, value in total_scores.items():
            print(f'{key}: {value}')

        # 返回所有计算出的分数
        return total_scores if total_scores else {"error": "No scores computed"}


if __name__ == "__main__":
    ref = {
        '1': ['go down the stairs and stop at the bottom .'],
        '2': ['this is a cat.']
    }
    gt = {
        '1': ['Walk down the steps and stop at the bottom. ', 'Go down the stairs and wait at the bottom.',
              'Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.'],
        '2': ['It is a cat.', 'There is a cat over there.', 'cat over there.']
    }

    Score = Scorer(ref, gt)
    Score.compute_scores()
