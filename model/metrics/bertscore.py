from typing import Dict, Sequence

from torchmetrics import Metric

from tokenizer import ptbtokenizer
import bert_score


class BERTScore(Metric):
    def __init__(self, **kwargs):
        super().__init__()
        self.tokenizer = ptbtokenizer.PTBTokenizer()
        self.scorer = bert_score.BERTScorer(**kwargs)
        self.add_state("gts", default=[], dist_reduce_fx="cat")
        self.add_state("gen", default=[], dist_reduce_fx="cat")
        self.add_state("ids", default=[], dist_reduce_fx="cat")
        self.eval, self.evalImgs = {}, {}

    def update(self, preds: Dict[int, Sequence[str]], target: Dict[int, Sequence[str]]):
        self.ids.extend(target.keys())
        self.gts.extend(target.values())
        self.gen.extend(preds.values())

    def compute(self):
        gts = {int(id): gts for id, gts in zip(self.ids, self.gts)}
        gen = {int(id): gen for id, gen in zip(self.ids, self.gen)}
        gts = self.tokenizer.tokenize(gts)
        gen = self.tokenizer.tokenize(gen)

        p, r, f1 = self.scorer.score([v[0] for v in gen.values()], [v for v in gts.values()])
        self.setEval({
            "precision": p.mean().item(),
            "recall": r.mean().item(),
            "f1": f1.mean().item()
        })
        # self.setEvalImgs()
        return self.eval

    def setEval(self, score):
        self.eval = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
