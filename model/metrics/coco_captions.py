from typing import List, Union, Sequence

from torchmetrics import Metric
import torch

from bleu.bleu import Bleu
from cider.cider import Cider
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from spice.spice import Spice
from tokenizer.ptbtokenizer import PTBTokenizer

"""
Torchmetrics wrapper for pycocoevalcap.COCOEvalCap
"""


class CocoScore(Metric):
    def __init__(self):
        super().__init__()
        self.tokenizer = PTBTokenizer()
        self.metrics = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]
        self.add_state("gts", default=[], dist_reduce_fx="cat")
        self.add_state("gen", default=[], dist_reduce_fx="cat")
        self.eval, self.evalImgs = {}, {}

    def update(self, preds: Sequence[str], target: Sequence[Sequence[str]]):
        self.gts.extend(target)
        self.gen.extend(preds)

    def compute(self):
        # print(self.gts, self.gen)
        gts = {i: [t] for i, t in enumerate(self.gts)}
        gen = {i: [p] for i, p in enumerate(self.gen)}
        for scorer, method in self.metrics:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, gen)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
            else:
                self.setEval(score, method)
            # self.setEvalImgs()
            print(score, scores)
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
