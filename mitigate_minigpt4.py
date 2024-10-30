import argparse
import random
import json
import random
from tqdm import tqdm
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, CONV_VISION_minigptv2, StoppingCriteriaSub, Conversation, SeparatorStyle
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

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

CONV_NOVISION_Vicuna0 = Conversation(
    system="Please answer my questions.",
    roles=("Human: ", "Assistant: "),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # llama
    #args = parser.parse_args('--cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0'.split())
    # vicuna
    args = parser.parse_args('--cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0'.split())
    return args

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]


vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = [StoppingCriteriaSub(stops=stop_words_ids)]

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)
print('Initialization Finished')



ds = json.load(open('result_minigpt_m2.json'))
ls = list(ds.keys())

# mitigate
chat_state = CONV_VISION.copy()
minigpt4_mllm_answers = dict()
for k in tqdm(ls):
    instances = ds[k]['instance']
    for idx, inst in enumerate(instances[::-1]):
        chat_state = CONV_VISION.copy()
        response='\u200b'
        while(response=='\u200b'):
            # eye on
            image_path = '/home/dycpu4_data1/ttchungac/MiniGPT-4/usage_figures/' + k + '.jpg'
            instruct_image = Image.open(image_path).convert("RGB")
            minigpt4_mllm_answers[k] = []
            img_list = []
            chat_state.messages = []
            chat.upload_img(image_path, chat_state, img_list)
            
            image_chat_state = chat_state.copy()
            chat.encode_img(img_list)
            questions = CAP_PROMPT.format(inst['question'])
            references = inst['answer']
            chat.ask(questions, chat_state)
            response = chat.answer(conv=chat_state,
                                        img_list=img_list,
                                        num_beams=5,
                                        temperature=1.0,
                                        repetition_penalty=1.0,
                                        max_new_tokens=300,
                                        max_length=2000)[0]
            # eye closed
            # https://github.com/K-tang-mkv/MiniGPT-4/blob/main/minigpt4/conversation/conversation.py
            chat_state2 = CONV_NOVISION_Vicuna0.copy()
            questions = LLM_GEN_PROMPT.format(response, inst['question'])
            references = inst['answer']
            chat_state2.append_message(chat_state2.roles[0], "<ImageHere>")
            chat.ask(questions, chat_state2)
            #img_list
            response = chat.answer(conv=chat_state2,
                            img_list=None,
                            num_beams=5,
                            temperature=1.0,
                            repetition_penalty=1.0,
                            max_new_tokens=500,
                            max_length=2000)[0]
        ds[k]['instance'][idx]['minigpt-4-7b(vicuna)_mitigated'] = response

json.dump(ds, open('result_minigpt_m2.json','w'))