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

## Run the Experiments

### Obtain LVLM's response.
The first step of the experiment is to obtain an LVLM's response of all the questions. We suggest you directly store the LVLM's answer as an item after each "question" item in the "tri-he.json" file. This helps you better align your results with the following codes. For example, you can store it with the key `xxx_triplets`, where `xxx` is the name of the evaluated LVLM.

### Extract Triplets from LVLM's response.
Next, we extract triplets from LVLM's response using GPT-4, which can be done by running the following.

```
python extract_triplet.py
```
Note that you need an access to the OpenAI's API with you own openai key, which should be added in line 63 of `bschecker/utils.py`. The extracted triplets will be stored in `'xxx_with_triplets.json` with the `xxx_triplets` key.

### Obtain hallucination judgement.

Finally, we can use both GPT-4 judge and NLI judge to obtain hallucination judgements of the extracted triplets.

To use GPT-4 judge, run
```
python gpt4_judge.py
```
where openai key is also needed.

To use NLI judge, run
```
python NLI_judge.py --model xxx
```

### Evaluation
To calculate the Hallu-I and Hallu-Q scores of different types of hallucinations, run
```
cd evaluation
python main_evaluation.py
```
### Obtain mitigation results via self-alignment.
To obtain a result with `Triplet Description` in the LLaVA-1.5 model, run
```
python mitigate_llava15.py
```

To obtain results with `Triplet Description` in the MiniGPT-4 model, run
```
python mitigate_minigpt4.py
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
