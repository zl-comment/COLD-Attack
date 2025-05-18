from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class SelfBleuScore:
    def __init__(self, weights=(0.25,0.25,0.25,0.25), smoothing=True):
        self.weights = weights
        self.smoothing = smoothing
        if smoothing:
            self.smoother = SmoothingFunction()
            self.smooth_fn = self.smoother.method1

    def compute_score(self, gts, res=None):
        """
        gts: dict[id] -> list of reference sentences （这里当做多候选）
        res: 不再使用，兼容现有接口
        返回:
          avg_self_bleu: 整体平均
          per_image_scores: list[Self-BLEU]，对应每个 id
        """
        per_image_scores = []
        for idx, hypos in gts.items():
            # 至少需要 2 条句子才能算 diversity
            if len(hypos) < 2:
                per_image_scores.append(0.0)
                continue

            scores = []
            for i, hyp in enumerate(hypos):
                # 取出除自己以外的所有句子作为参考
                refs = [h.split() for j, h in enumerate(hypos) if j != i]
                hyp_tokens = hyp.split()
                score_i = sentence_bleu(
                    refs,
                    hyp_tokens,
                    weights=self.weights,
                    smoothing_function=(self.smooth_fn if self.smoothing else None)
                )
                scores.append(score_i)

            # 该 id 下的平均 Self-BLEU
            per_image_scores.append(sum(scores) / len(scores))

        avg_self_bleu = sum(per_image_scores) / len(per_image_scores)
        return avg_self_bleu, per_image_scores

    def method(self):
        return "Self-BLEU"
