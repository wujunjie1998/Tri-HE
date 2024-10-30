import pdb
from typing import List

from .checker_base import CheckerBase
from ..utils import get_openai_model_response


GPT4_CHECKING_PROMPT_Q = \
"""I have a claim that made by a language model to a question, please help me for checking whether the claim can be entailed according to the provided reference which is related to the question. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If the claim is contradicted with the reference, answer 'Contradiction'.
If the reference is not relevant to the claim or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Question:
{question}

### Reference:
{reference}

### Claim:
{claim}

Your answer should be only a single word in ['Entailment', 'Neutral', 'Contradiction']
"""

GPT4_CHECKING_PROMPT = \
"""Please help me check whether a claim, represented as a triplet ('subject', 'predicate', 'object'), is supported by a list of reference triplets. The reference is a list of triplets formatted as ('subject', 'predicate', 'object').

Your task is to determine if the claim is directly supported by any single triplet in the reference or can be reasonably inferred from a combination of multiple triplets. When evaluating the claim:

Direct Support: Answer 'yes' if the exact triplet appears in the reference list.
Inference: Answer 'yes' if the claim can be logically inferred from one or more triplets in the reference. Pay special attention to:
General Inferences: Consider common associations or implications (e.g., green leaves typically imply spring).
Conditional Phrases: Pay attention to phrases like 'could be', 'might', 'suggests', which allow for broader inferences.
Not Supported: Answer 'no' if the claim neither directly matches any triplet in the reference nor can be reasonably inferred.
Please avoid using your own external knowledge. Base your judgement solely on the information provided in the reference triplets and the claim.

### Reference:
{reference}

### Claim:
{claim}

Your answer should be only a single word in ['yes', 'no']
"""


class GPT4Checker(CheckerBase):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_temp = GPT4_CHECKING_PROMPT
        self.prompt_temp_wq = GPT4_CHECKING_PROMPT_Q

    def _check(
        self, 
        claims: List,
        references: List,
        response: str,
        question: str, 
    ):
        ret_labels = []
        #for claim, reference in zip(claims, references):
        #if isinstance(claim, list):
            #assert len(claim) == 3
            #claim = f"({claim[0]}, {claim[1]}, {claim[2]})"
        if question is None:
            prompt = self.prompt_temp.format(
                reference=references,
                claim=tuple(claims)
            )
        else:
            prompt = self.prompt_temp_wq.format(
                question=question,
                reference=references,
                claim=tuple(claims)
            )

        openai_response = get_openai_model_response(
            prompt=prompt,
            temperature=0,
            model='gpt-4'
        )
        if openai_response and len(openai_response):
            label = None
            #if self.label_contradiction.lower() in openai_response.lower():
                #label = self.label_contradiction
            if self.label_entailment.lower() in openai_response.lower():
                label = self.label_entailment
            else:
                label = self.label_neutral
            ret_labels.append(label)
        else:
            raise 'OpenAI API returns None or empty string'
        return ret_labels
