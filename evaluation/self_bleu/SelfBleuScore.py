from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class SelfBleuScore:
    def __init__(self, weights=(0.25,)*4, smoothing=True):
        self.weights = weights
        self.smoothing = smoothing
        if smoothing:
            self.smoother  = SmoothingFunction()
            self.smooth_fn = self.smoother.method1

    def compute_score(self, gts, res=None):
        """
        gts: dict[id] -> list[str] 多条句子
        返回 avg_self_bleu, per_image_self_bleu
        """
        per_scores = []
        for idx, hypos in gts.items():
            # 少于2条无法计算多样性
            if len(hypos) < 2:
                per_scores.append(0.0)
                continue

            scores = []
            for i, hyp in enumerate(hypos):
                refs = [h.split() for j,h in enumerate(hypos) if j!=i]
                hyp_toks = hyp.split()
                sc = sentence_bleu(
                    refs,
                    hyp_toks,
                    weights=self.weights,
                    smoothing_function=(self.smooth_fn if self.smoothing else None)
                )
                scores.append(sc)
            per_scores.append(sum(scores)/len(scores))

        avg = sum(per_scores) / len(per_scores)
        return avg, per_scores

    def method(self):
        return "Self-BLEU"
