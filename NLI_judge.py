import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
from tqdm import tqdm

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--compression_model_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_token_len", help='max token length for ICL demos', default=750)
    parser.add_argument("--compression_rate", default=0.1)
    parser.add_argument("--target_device", default='cuda')
    parser.add_argument("--compression_device", default='cpu')
    args = parser.parse_args()
    return args

def runkey(subtree, k, printres=True, modeleval=MODEL+'_triplets', yesc=[], noc=[], printjudge=False, judgen=None):
    lsofres1=[]
    triplets = [t.replace('(','').replace(')','').replace(',','')+'.' for t in subtree[k]['triplets']]
    embeddings = model.encode(triplets)
    if(printres): print(' '.join(triplets),'\n')
    c = 0
    pbar = tqdm(range(len(subtree[k]['instance'])))
    for i in pbar:
        lsofres1.append([])
        if('triplets' not in modeleval): 
            if('sentences' in subtree[k]['instance'][i].keys()):
                evaltriplets = subtree[k]['instance'][i]['sentences']
            else:
                evaltriplets = subtree[k]['instance'][i][modeleval].split('. ')
                evaltriplets = [e if e[-1]=='.' else e+'.' for e in evaltriplets]
                subtree[k]['instance'][i]['sentences'] = evaltriplets
        else: evaltriplets = subtree[k]['instance'][i][modeleval]
            #objectsent = 'There are '+', '.join(list(set(tree['minigpt4'][k]['object']).union(set(tree['minigpt4'][k]['all_object']))))+'.'
        for tdx,t in enumerate(evaltriplets):
            
            if('triplets' not in modeleval): t = [t]
            if(printres): print(c, [' '.join(t)])
            c+=1
            src = model.encode([' '.join(t)])
            res = cosine_similarity(src,embeddings)
            # filter not useful triplets
            filtered = [' '.join([triplets[idx] for idx in np.nonzero(res>0.5)[1]])]
            if(filtered==['']): 
                oriembeddings = model.encode([' '.join(triplets)])
                topk = (-res).argsort()[0,:3]
                if(printres): print('filtered',[triplets[tdx] for tdx in topk])
                oriembeddings = model.encode([' '.join([triplets[tdx] for tdx in topk])])
                entailval = cosine_similarity(src,oriembeddings)[0][0]
                #newembeddings = model.encode(filtered)
                #entailval = cosine_similarity(src,newembeddings)[0][0]
            else: 
                if(printres): print('filtered',filtered)
                oriembeddings = model.encode([' '.join(triplets)])
                topk = (-res).argsort()[0,:3]
                oriembeddings = model.encode([' '.join([triplets[tdx] for tdx in topk])])
                orival = cosine_similarity(src,oriembeddings)[0][0]
                newembeddings = model.encode(filtered)
                entailval = cosine_similarity(src,newembeddings)[0][0]
                entailval = max(entailval, orival)
            if(entailval>0.6): 
                if(printres): print(entailval,'yes')
                lsofres1[-1].append('yes')
            else: 
                if(printres): print(entailval,'no')
                lsofres1[-1].append('no')
            if(printres): 
                if(printjudge and judgen in subtree[k]['instance'][i].keys() and tdx<len(subtree[k]['instance'][i][judgen])): print(subtree[k]['instance'][i][judgen][tdx])
                print()
        if(judgen!=None): subtree[k]['instance'][i][judgen+'_nli'] = lsofres1[-1] #add new keys of nli judgement
        lsofresall = []
        for l in lsofres1:
            lsofresall += l
        yes_c, no_c = len([r for r in lsofresall if r=='yes']), len([r for r in lsofresall if r=='no'])
        pbar.set_postfix({'entail': (yes_c+sum(yesc))/(yes_c+sum(yesc)+no_c+sum(noc))})
    return lsofres1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with flags and arguments.")
    parser.add_argument('--model', help='model for evaluation')
    args = parser.parse_args()
    MODEL = args.model
    
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    tree = {
        MODEL: json.load(open(MODEL+'_with_triplets.json')),
    }
    
    ks = list(tree[MODEL].keys())
    
    treens = list(tree.keys())
    modelevalns = [MODEL+'_triplets']
    judgens = [MODEL+'_triplets_judgements']
    
    # NLIeval on triplets
    for x in range(len(treens)):
        yesc, noc = [], []
        for k in ks:
            # print('key:',k)
            res = runkey(tree[treens[x]], k, printres=False, modeleval=modelevalns[x], yesc=yesc, noc=noc, printjudge=True, judgen=judgens[x])
    print('The result of NLI Judgement is updated in the json file.')

########### topn ablation ###########
# from nltk.tokenize.treebank import TreebankWordDetokenizer
# modeln = ['shikra', 'internlm', 'instructblip', 'llava15', 'llava1', 'minigpt4']
# modelans = ['shikra-7b', 'InternLM_XComposer2_VL', 'instruct_blip_7b', 'llava-1.5-7b', 'llava-1.1-7b', 'minigpt-4-7b(vicuna)']
# tripletsn = ['shikra_7b_triplets', 'internlm_triplets', 'instruct_blip_7b_triplets', 'llava-1.5_7b_triplets', 'llava-1.1_7b_triplets', 'minigpt4_7b(vicuna)_triplets']

# from nltk.tokenize import word_tokenize
# for no in range(len(modeln)):
#     for k in tqdm(ks):
#         for idx,i in enumerate(tree[modeln[no]][k]['instance']):
#             maxtop50 = []
#             tokenized = word_tokenize(i[modelans[no]])
#             #print(tokenized)
#             for topw in topls:
#                 b50, a50 = TreebankWordDetokenizer().detokenize(tokenized[:topw]), TreebankWordDetokenizer().detokenize(tokenized[topw:])
#                 eb50, ea50 = model.encode(b50.split('.')), model.encode(a50.split('.'))
#                 #print(b50,'\n',a50)
#                 top50triplets = []
#                 for jdx,j in enumerate(i[tripletsn[no]]):
#                     t = ' '.join(j)+'.'
#                     t = model.encode([t])
#                     if(cosine_similarity(t,eb50).max()<cosine_similarity(t,ea50).max()):
#                         break
#                     top50triplets.append(j)
#                 if(len(top50triplets)>len(maxtop50)): maxtop50 = top50triplets
#                 tree[modeln[no]][k]['instance'][idx][tripletsn[no]+'top'+str(topw)+'w'] = maxtop50
#                 tree[modeln[no]][k]['instance'][idx][modelans[no]+'top'+str(topw)+'w'] = b50
