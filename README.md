# Tri-HE

[![arXiv](https://img.shields.io/badge/arXiv-2410.23114-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2410.23114) [![Web](https://img.shields.io/badge/Web-Tri_HE-blue.svg?style=plastic)](https://kaichen1998.github.io/projects/tri-he/)

This repository contains the implementation of the paper:

> Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models <br>
> [Junjie Wu](https://wujunjie1998.github.io/)\*, [Tsz Ting Chung](https://ttchungc.github.io/)\*, [Kai Chen](https://kaichen1998.github.io)\*, [Dit-Yan Yeung](https://sites.google.com/view/dyyeung) <br>
> *Equal Contribution


## Tri-HE Benchmark Data

The Tri-HE benchmark data contains 300 GQA images, each quipped with several elaborately crafted questions, along with reference triplets and objects, enabling unified assessment of LVLM's hallucination.

- **Data Structure**: Each entry has a unique ID as the key of the image (e.g., `"2374892"`), containing the following main elements:
  - **`instance`**: A list of question-answer pairs. Each item in this list contains:
    - `question`: A question that is related to the given image.
    - `answer`: The ground truth answer of the question.
    - `triplet`: Triplets that support the ground truth answer. 
  - **`triplets`**: All triplets that are related to the given image.

  - **`all_object`**: All objects identified in the image.

  - **`object`**: A subset of key objects or entities directly related to the question-answer pairs and triplet relations (e.g., `["laptop", "illuminating room", "pen"]`).

### Example Entry

An example data entry for a single instance is as follows:

```json
{
    "2374892": {
        "instance": [
            {
                "question": "Why is the room possibly illuminated?",
                "answer": "The room is possibly illuminated because there is a lamp turned on that is lighting the area.",
                "triplet": ["(Lamp, turned on, illuminating room)"],
            },
            ...
        ],
        "triplets": [
            "(laptop, is, black)",
            "(shirt, is, white)",
            "(Lamp, turned on, illuminating room)",
            ...
        ],
        "all_object": [
            "laptop", "screen", "hand", "pen", "lamp", "desk", "man", ...
        ],
        "object": [
            "laptop", "illuminating room", "pen", "desk", "man", ...
        ]
    }
}
```

### Images
All the 300 images are stored in `data/usage_figures`.

## Example Usage of NLI_judge.py
With formatted `minigpt4_with_triplets.json`, running the below command store the NLI judgement result in the json.
```
python NLI_judge.py --model minigpt4
```

## Citation

```bibtex
@article{wu2024unified,
  title={Unified Triplet-Level Hallucination Evaluation for Large Vision-Language Models},
  author={Wu, Junjie and Chung, Tsz Ting and Chen, Kai and Yeung, Dit-Yan},
  journal={arXiv preprint arXiv:2410.23114},
  year={2024}
}
```
