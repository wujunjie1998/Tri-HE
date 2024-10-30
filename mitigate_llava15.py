import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import json

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


from PIL import Image
import math

CAP_PROMPT = (
    'REQUEST:\n{}\n\nBased on the REQUEST, identify the objects that are CERTAINLY PRESENTED in the provided image, and describe the relationships between the identified objects.'
)

CAP_GENERAL_PROMPT = (
    'REQUEST:\n{}\n\nBased on the REQUEST, describe the image.'
)

LLM_GEN_PROMPT = (
    "You are given some hints regarding a question on an image.\n\n"
    "Hints: \"Answer: {}\"\n\n"
    "Based on the hints, answer the following question WITHOUT HALLUCINATION.\n\n"
    "{}"
)

def wrap_and_generate(query, model, tokenizer, image=None):
    temperature = 0.2
    top_p = 0.7
    num_beams = 1
    # cases that use images
    if image is not None:
        if model.config.mm_use_im_start_end:
            query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + query
        else:
            query = DEFAULT_IMAGE_TOKEN + '\n' + query
            
    conv = conv_templates['vicuna_v1'].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image.unsqueeze(0).half().cuda() if image is not None else None,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs

ds = json.load(open('llava-1.5_7b_300_triplets_new.json'))
ls = list(ds.keys())

for k in tqdm(ls):
    image_path = 'usage_figures/' + k + '.jpg'
    image = Image.open(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    for idx, i in enumerate(ds[k]['instance']):
        qs = i['question']
        ######################################### CAP-START #########################################
        cap_qs = CAP_PROMPT.format(qs)
        caption = wrap_and_generate(cap_qs, model, tokenizer, image=image_tensor)
        
        ######################################### LLM-GEN-START #########################################
        gen_qs = LLM_GEN_PROMPT.format(caption, qs)
        safe_answer = wrap_and_generate(gen_qs, model, tokenizer, image=None)
        ds[k]['instance'][idx]['llava-1.5-7b_mitigated'] = safe_answer

json.dump(ds, open('llava-1.5_7b_300_triplets_mitigated.json','w'))