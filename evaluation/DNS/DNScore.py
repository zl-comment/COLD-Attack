from distinct_n import distinct_n_corpus_level


class DNScore:
    def __init__(self):
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):
        # 将所有句子合并为一个列表，计算语料库级别的 Distinct-1 和 Distinct-2 分数
        all_sentences = []
        for key, gt_sentences in gts.items():
            all_sentences.extend(gt_sentences)  # 合并所有句子

        # 计算整个语料库的 Distinct-1 和 Distinct-2 分数
        distinct_corpus_1 = distinct_n_corpus_level(all_sentences, 1)
        distinct_corpus_2 = distinct_n_corpus_level(all_sentences, 2)

        # 返回计算的 DNS 分数
        return distinct_corpus_1 ,distinct_corpus_2


    def method(self):
        return "DNS"

