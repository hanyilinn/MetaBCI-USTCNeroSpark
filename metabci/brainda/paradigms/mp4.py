"""
Motor Imagery Paradigm.

"""
from .base import BaseParadigm


class Video(BaseParadigm):
    """Basic motor imagery paradigm."""

    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != "mp4":
            ret = False
        return ret