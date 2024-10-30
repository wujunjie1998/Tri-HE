from typing import List, Union
import pdb
from ..utils import split_text


def merge_ret(ret):
    """Merge results from multiple paragraphs"""
    if "yes" in ret:
        return "yes"
    if "no" in ret:
        return "no"
    return "Neutral"


def merge_multi_psg_ret(ret):
    """Merge results from multiple passages
    TODO: consider possible cases where the results are inconsistent.
    """
    if "yes" in ret:
        return "yes"
    if "no" in ret:
        return "no"
    return "Neutral"


class CheckerBase:
    def __init__(self) -> None:
        self.label_entailment = 'yes'
        self.label_neutral = 'no'
        #self.label_contradiction = 'Contradiction'
        self.labels = ["yes", "no"]

    def check(
        self, 
        claim: str, 
        reference: Union[str, List], 
        response: str = None,
        question: str = None,
        max_reference_segment_length: int = 200, 
    ):
        ret = []
        if isinstance(reference, str):
            reference = [reference]
        #for psg in reference:
            '''
            if max_reference_segment_length > 0:
                segments = split_text(psg, max_reference_segment_length)
            else:
                segments = [psg]
            '''
            #segments = psg

        psg_ret = self._check(
            claims=claim,
            references=reference,
            response=response,
            question=question,
        )

        ret.append(merge_ret(psg_ret))

        return merge_multi_psg_ret(ret)

    def _check(
        self,
        claims: List,
        references: List,
        response: str,
        question: str = None
    ):
        raise NotImplementedError
