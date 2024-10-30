from openai import OpenAI
import openai
from tqdm import tqdm
import pdb
import time
import os
import shutil
import copy
import tiktoken
import base64
import requests
import json
import ast

## Load triplet judgement results
## In the below codes, we use the judgements from gpt-4 judge as an example.
## If you want to evaluate NLI's results, you can simply change 'xxx_triplets_judgements' to 'xxx_triplets_judgements_nli'.

with open('judgements/xxx_triplets_results.json', 'r') as file:
    lvlm = json.load(file)

## Hallu-Q
lvlm_score = []
for i, index in enumerate(lvlm.keys()):
    for j, instance in enumerate(lvlm[index]['instance']):
        try:
            lvlm_score.append(instance['xxx_triplets_judgements'].count('no') / len(instance['xxx_triplets_judgements']))
        except ZeroDivisionError:
            continue
        except KeyError:
            continue

print("Hallu-Q: ", sum(lvlm_score)/len(lvlm_score))

## Hallu-I
lvlm_scores = []
for i, index in enumerate(lvlm.keys()):

    lvlm_score = []
    for j, instance in enumerate(lvlm[index]['instance']):
        try:
            lvlm_score.append(instance['xxx_triplets_judgements'].count('no') / len(instance['xxx_triplets_judgements']))
        except ZeroDivisionError:
            continue
    try:
        lvlm_scores.append(sum(lvlm_score)/len(lvlm_score))
    except ZeroDivisionError:
        continue

print("Hallu-I: ", sum(lvlm_scores)/len(lvlm_scores))

## Object/Relation Hallucination
with open('responses/xxx_triplets_results.json', 'r') as file:
    lvlm_judge = json.load(file)

## Hallu-Q
lvlm_object_scores = []
lvlm_relation_scores = []
for i, index in enumerate(lvlm_judge.keys()):
    for j, judgements in enumerate(lvlm_judge[index]):
        objects = 0
        relations = 0
        for judgement in judgements:
            if ("my answer is 'no'" in judgement.lower()) or ("my answer is \"no\"" in judgement.lower()):
                if ("the error is related to 'object1'" in judgement.lower()) or ("the error is related to 'object2'" in judgement.lower()):
                    objects += 1
                else:
                    relations += 1
        try:
            lvlm_object_scores.append(objects/len(judgements))
            lvlm_relation_scores.append(relations/len(judgements))
        except ZeroDivisionError:
            continue

print('Object Hallu-Q: ', sum(lvlm_object_scores)/len(lvlm_object_scores))
print('Relation Hallu-Q: ', sum(lvlm_relation_scores)/len(lvlm_relation_scores))

## Hallu-I
lvlm_object_scores = []
lvlm_relation_scores = []
for i, index in enumerate(lvlm_judge.keys()):
    lvlm_object_score = []
    lvlm_relation_score = []
    for j, judgements in enumerate(lvlm_judge[index]):
        objects = 0
        relations = 0

        for judgement in judgements:
            if ("my answer is 'no'" in judgement.lower()) or ("my answer is \"no\"" in judgement.lower()):
                if ("the error is related to 'object1'" in judgement.lower()) or (
                        "the error is related to 'object2'" in judgement.lower()):
                    objects += 1
                else:
                    relations += 1
        try:
            lvlm_object_score.append(objects / len(judgements))
            lvlm_relation_score.append(relations / len(judgements))
        except ZeroDivisionError:
            continue
    try:
        lvlm_object_scores.append(sum(lvlm_object_score) / len(lvlm_object_score))
        lvlm_relation_scores.append(sum(lvlm_relation_score) / len(lvlm_relation_score))
    except ZeroDivisionError:
        continue

print('Object Hallu-I: ', sum(lvlm_object_scores)/len(lvlm_object_scores))
print('Relation Hallu-I: ', sum(lvlm_relation_scores)/len(lvlm_relation_scores))



















