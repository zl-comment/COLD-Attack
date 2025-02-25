import torch
import numpy as np
import time
import wandb

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
class Scorer():
    def __init__(self,ref,gt):
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
    
    def compute_scores(self):
        total_scores = {}
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
            print('{}:{}'.format(key, value))
        
        # 确保返回total_scores
        return total_scores if total_scores else {"error": "No scores computed"}


if __name__ == "__main__":
    # rouge = Rouge()
    # bleu = Bleu()
    # cider = Cider()
    # Meteor = Meteor()
    # Bertscore = BertScore()

    ref = {
        '1':['go down the stairs and stop at the bottom .'],
        '2':['this is a cat.']
    }
    gt = {
        '1':['Walk down the steps and stop at the bottom. ', 'Go down the stairs and wait at the bottom.','Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.'],
        '2':['It is a cat.','There is a cat over there.','cat over there.']
    }
    Score = Scorer(ref, gt)
    # print(calculate_rouge(output, original))
    # print(calculate_bleu_score(output, original))
    # print(bleu.compute_score(gt, ref))
    # print(rouge.compute_score(gt, ref))
    Score.compute_scores()