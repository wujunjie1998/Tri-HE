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

## Import OpenAI Key
openai.organization = "your openai organization"
chatgpt_client = OpenAI(api_key='your openai key')


## Import data
with open('xxx_with_triplets.json', 'r') as file:
    triplets = json.load(file)


## Instructions for GPT-4 evaluation
evaluate_instruction = 'Given a list of reference triplets ("object1", "relation", "object2") extracted from the scene graph of an image, along with a list of objects observed in this image, your task is:\n\n' \
'Task 1. Determine if a claim triplet ("object1", "relation", "object2") is directly supported by any single triplet in the reference, or can be logically inferred from multiple reference triplets and the list of objects. Follow these steps when finishing the task:\n\n' \
'1. Answer "yes" if the claim appears in the reference.\n\n' \
'2. Answer "yes" if the claim can be logically inferred from one or more triplets in the reference. Consider:\n\n' \
'a. General Inferences: Assess common associations or implications.\n' \
'b. Conditional Phrases: Note phrases like "could be", "might", "suggests", which allow broader inferences.\n' \
'c. Equivalence of Objects: In your judgment, treat objects of the same kind as equal. For example, "woman", "man" should be considered under the general category of "person".\n' \
'd. Support from Object List: If the claim is not directly supported or inferable from the triplets, assess whether the list of objects provides additional evidence to support or infer the claim.\n\n' \
'3. Answer "no" if the claim neither directly matches any triplet in the reference nor can be reasonably inferred from the triplets and the object list.\n\n' \
'Task 2: Error categorization.\n\n' \
'If your answer to the previous task is "no", determine whether the not supported/inferred part in the claim is "object1" or "object2" or "relation".\n\n' \
'Reference:\n{}\n\n' \
'List of Objects:\n{}\n\n' \
'Claim:\n{}\n\n' \
'Please output your answer to the first task only in the format of "My answer is \'yes\'/\'no\'". If your answer is "no", output your answer to the second task only in the format of "The error is related to \'object1\'/\'object2\'/\'relation\'".'

questions_ids = ['2374892', '2397582', '2387230', '2383415', '2380479', '2402409', '2391748', '2404666', '2315880', '2331017', '2347569', '2319825', '2325480', '2407148',
                    '2402232', '2405227', '2359433', '2396963', '2333677', '2395042', '2363853', '2362053', '2386393', '2410230', '2337344', '2317575', '2367982', '1592403',
                    '2367184', '2395567', '2414858', '2406059', '2371404', '2379186', '2399534', '286053', '2380125', '2325937', '2395794', '2385456', '2317148', '2412131', '2335787',
                    '2331948', '2391000', '2397448', '2415899', '2341361', '2358708', '2323530', '2379961', '713360', '2402763', '2393510', '2320543', '2412046', '2380680', '2396106',
                    '2347091', '2409544', '2387777', '2337304', '2376529', '2353439', '3455', '2396483', '2390229', '2367328', '2406011', '2346625', '2365078', '2368472', '2393281',
                    '2326674', '2375491', '2384779', '2384092', '2366958', '2406666', '2410898', '2407164', '2355023', '2318090', '2386254', '2386334', '2404613', '2383991', '2341840',
                    '2365562', '2388962', '2391950', '2319558', '2389203', '2389205', '2409543', '2410392', '2319034', '2411516', '2355548', '2383328', '1592175', '2404536', '2411375',
                    '2412542', '2364025', '2320507', '2394753', '2376660', '2393164', '2380977', '2380375', '2362253', '314', '2393775', '2325932', '2362477', '2335178', '2328879',
                    '2341512', '2403003', '2381595', '2376620', '2337945', '2386867', '2401166', '2363959', '2404602', '2408596', '2355237', '2316628', '2329334', '2379906', '2397488',
                    '2359140', '2371945', '2411488', '2350736', '2404849', '2331876', '2401506', '2392665', '2406998', '2387508', '2391364', '2343796', '2415157', '2400560', '2385013',
                    '2342134', '2390108', '2348072', '2323054', '2397341', '2366876', '2390911', '2391205', '2390599', '2333858', '2861', '2391835', '2342333', '2323323', '2382847', '2405830',
                    '2347819', '2384812', '2368361', '2414129', '2272', '2336635', '2400789', '2405555', '2407415', '2323268', '2376766', '1591820', '2347171', '2396799', '2325882', '2320615',
                    '2399206', '2341698', '2364886', '2384012', '2369640', '2326613', '2403060', '2388598', '2355736', '2407347', '2327560', '2402334', '2357353', '2356008', '2387752',
                    '2406309', '2341341', '2323243', '2342451', '2372614', '2401249', '2378167', '2394962', '2352009', '2356494', '713531', '2346104', '2377608', '2393430', '2359715',
                    '2371761', '2374803', '2383613', '2383614', '2414693', '2353016', '2320407', '2406323', '2343415', '2412204', '2407451', '2383582', '2402165', '2316099', '70',
                    '2353731', '2322017', '2318853', '2343509', '2370948', '2380022', '2354914', '2356041', '2399053', '2348163', '1592683', '2352342', '2317627', '2386682', '2385133',
                    '2326672', '2403546', '2389793', '2320676', '2378619', '2373168', '2362327', '2400175', '2390700', '2385788', '1159264', '2398554', '2351181', '2375868', '2320339',
                    '2346506', '2339309', '2346094', '1592103', '1159845', '2400618', '2366184', '2400603', '2327311', '2371558', '2368487', '2406620', '2251', '2374729', '2340730', '2369531',
                    '2342152', '2389148', '2397079', '2393941', '2397910', '2366510', '2368283', '2375806', '2343787', '2405896', '2409391', '2378861', '3898', '2414630', '2387206', '2362421',
                    '2344585', '2319086', '2398388', '2367256', '2346063', '2331048', '2360708', '2401215', '2412723', '2315718', '2387283', '2322244', '2409876']


## Start to evaluate
model_responses = {}
for index in tqdm(triplets.keys()):
    reference = triplets[index]['triplets']
    object_list = triplets[index]['all_object']

    new_reference = [tuple(item.strip('()').split(', ')) for item in reference]
    model_response = []

    for i, instance in tqdm(enumerate(triplets[index]['instance'])):

        judgements = []
        ori_responses = []
        for claim in instance['xxx_triplets']:
            judge_prompt = copy.deepcopy(evaluate_instruction).format(new_reference, object_list,
                                                                      tuple(claim), '{}', '{}', '{}')

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": judge_prompt}]

            response = chatgpt_client.chat.completions.create(
                model="gpt-4-1106-preview",
                # response_format={ "type": "json_object" },
                messages=messages,
                # temperature=0.7,
            )

            ori_responses.append(response.choices[0].message.content)

            ## judge the result
            if ("my answer is 'yes'" in response.lower()) or (
                    "my answer is \"yes\"" in response.lower()):
                judgements.append('yes')
            elif ("my answer is 'no'" in response.lower()) or (
                    "my answer is \"no\"" in response.lower()):
                judgements.append('no')
            else:
                judgements.append('null')
        model_response.append(ori_responses)
        instance['xxx_triplets_judgements'] = judgements

    model_responses[index] = model_response
    ## Store the results
    if not os.path.exists('judgements'):
        os.makedirs('judgements')

    # 检查并创建'responses'目录
    if not os.path.exists('responses'):
        os.makedirs('responses')

    with open(
            'judgements/xxx_triplets_results.json',
            'w') as file:
        json.dump(triplets, file)

    with open(
            'responses/xxx_triplets_results.json',
            'w') as file:
        json.dump(model_responses, file)
