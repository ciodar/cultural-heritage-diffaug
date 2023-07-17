from typing import List, Union, Sequence, Dict

from torchmetrics import Metric

from bleu.bleu import Bleu
from cider.cider import Cider
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from spice.spice import Spice
from tokenizer import ptbtokenizer

"""
Torchmetrics wrapper for pycocoevalcap.COCOEvalCap
"""


class CocoScore(Metric):
    def __init__(self, **kwargs):
        super().__init__()
        self.tokenizer = ptbtokenizer.PTBTokenizer()
        self.metrics = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]
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
        for scorer, method in self.metrics:
            # print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, gen)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
            else:
                self.setEval(score, method)
            # self.setEvalImgs()
        return self.eval

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
