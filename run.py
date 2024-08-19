from interpolation_models import HFInstructModelInterpolation, ChatGPTInterpolation
from translation_models import GPTTranslator
gpt_translator = GPTTranslator()
import pandas as pd
import json
from tqdm import tqdm 
import random
random.seed(42)
from collections import Counter

TEST_SETTINGS = {
    'mt_model':'gpt-3.5-turbo-0125',
    'intp_model':'Qwen/Qwen2-72B-Instruct',
    'start_select_strategy':'sort-S-T', #random, sbert_filter, sort, tops
    'starts_num':3, #3, 5, 10
    'aggregate_method':'poll', # none, prompt, poll
    'lang':'ko'
}

LANG_MAP = {
    'ko':'Korean',
    'zh':'Chinese',
    'sw':'Swahili',
    'be':'Bengali',
    'mr':'Marathi'
}

gpt_interpolator = ChatGPTInterpolation()
hf_interpolator = HFInstructModelInterpolation(model_id=TEST_SETTINGS['intp_model'], quantization=True)

def get_sst_prop_name(symbol):
    if(symbol == 'S'):
        return ('sbert_sim','desc')
    elif(symbol == 'L'):
        return ('lev_dist','asc')
    elif(symbol == 'T'):
        return ('tree_edit_dist','asc')

def get_sort_key(criteria):
    def sort_key(item):
        return tuple(
            (item[attribute] if order == 'asc' else -item[attribute])
            for attribute, order in criteria
        )
    return sort_key

def get_start_id_list(end_id, sim_scores, start_select_strategy, starts_num):

    def get_unique_top_1(sorted_list, unique_set):
        for element in sorted_list:
            if element['id'] not in unique_set:
                unique_set.add(element['id'])
                return element
        return None

    sst =  start_select_strategy.split('-')
    if(sst[0] == 'random'):
        sim_score = sim_scores[str(end_id)]
        sim_score = random.sample(sim_score,k=starts_num)
        start_id_list = [x["id"] for x in sim_score]
    elif(sst[0] == "sort"):
        criteria = [get_sst_prop_name(x) for x in sst[1:]]
        sim_score = sim_scores[str(end_id)]
        sim_score = sorted(sim_score, key=get_sort_key(criteria))
        sim_score = sim_score[:starts_num]
        start_id_list = [x["id"] for x in sim_score]
    elif(sst[0] == "sbert_filter"):
        sim_score = sim_scores[str(end_id)]
        sim_score = sorted(sim_score, key=lambda x:x['sbert_sim'], reverse=True)
        sim_score = sim_score[:10]
        criteria = [get_sst_prop_name(x) for x in sst[1:]]
        sim_score = sorted(sim_score, key=get_sort_key(criteria))
        sim_score = sim_score[:starts_num]
        start_id_list = [x["id"] for x in sim_score]
    else:
        unique_elements = set()
        sim_score = sim_scores[str(end_id)]
        sorted_by_sbert = sorted(sim_score, key=lambda x:x['sbert_sim'], reverse=True)
        top_1_id_by_sbert = get_unique_top_1(sorted_by_sbert, unique_elements)

        sorted_by_ted = sorted(sim_score, key=lambda x:x['tree_edit_dist'])
        top_1_id_by_ted = get_unique_top_1(sorted_by_ted, unique_elements)

        sorted_by_lev = sorted(sim_score, key=lambda x:x['lev_dist'])
        top_1_id_by_lev = get_unique_top_1(sorted_by_lev, unique_elements)

        start_id_list = [e["id"] for e in [top_1_id_by_sbert, top_1_id_by_ted, top_1_id_by_lev] if e is not None]
    return start_id_list

#Load Test Dataset
test_df = pd.read_csv('data/flores_test.csv',encoding='utf-8')

#Load Similarity Scores
with open(f"data/sim_scores_{TEST_SETTINGS['lang']}.json", 'r', encoding='utf-8') as f:
    sim_scores = json.load(f)

#Load Interpolation and MT data
with open(f"data/intp_mt_{TEST_SETTINGS['lang']}.json",'r', encoding='utf-8') as f:
    intp_n_mt = json.load(f)

#Load Start Sents
start_sent_df = pd.read_csv(f"data/start_sent_pool_{TEST_SETTINGS['lang']}.csv", encoding='utf-8')

#Make Test Cache
test_cache = {'mt_model':TEST_SETTINGS['mt_model'], 'intp_model':TEST_SETTINGS['intp_model']}
for i, end in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    #Search for start sents
    ##Sort by criteria
    start_id_list = get_start_id_list(end['id'], sim_scores, TEST_SETTINGS['start_select_strategy'],TEST_SETTINGS['starts_num'])
    
    #Iterate over sorted start sents
    test_cache[end['id']] = {
        'src':end['en'],
        'ref':end[TEST_SETTINGS['lang']],
        'start_id': start_id_list,
        'mts': []
    }

    for start_id in start_id_list: 
        key = f'{start_id}-{end["id"]}'
        start_sent_data = start_sent_df[start_sent_df['id'] == start_id].iloc[0]
        start_src = start_sent_data['src']
        start_mt = start_sent_data['mt']
        start_ref = start_sent_data['ref']

        if((key in intp_n_mt) and (TEST_SETTINGS['intp_model'] in intp_n_mt[key])): #Check if interpolation data exists
            intp_sents = intp_n_mt[key][TEST_SETTINGS['intp_model']]['intp']
        else:
            if(TEST_SETTINGS['intp_model'].startswith("gpt")):
                intp_sents = gpt_interpolator.interpolate(start_src,end['en'])
            else:
                intp_sents = hf_interpolator.interpolate(start_src,end['en'])

        if(key not in intp_n_mt): #Add dict if there is no key in intp data
            intp_n_mt[key] = {}
        if(TEST_SETTINGS['intp_model'] not in intp_n_mt[key]):
            intp_n_mt[key][TEST_SETTINGS['intp_model']] = {}

        if(intp_sents[0]!='ERROR'): #Check for interpolation Error
            if("mt" not in intp_n_mt[key][TEST_SETTINGS['intp_model']]):    
                end_mt, grad_mts = gpt_translator.grad_translation(start_src, start_mt, end['en'], intp_sents, target_lang=LANG_MAP[TEST_SETTINGS['lang']], max_retries=3)
                test_cache[end['id']]['mts'].append(end_mt)
            else:
                test_cache[end['id']]['mts'].append(intp_n_mt[key][TEST_SETTINGS['intp_model']]['mt'])
                end_mt = intp_n_mt[key][TEST_SETTINGS['intp_model']]['mt']
                grad_mts = intp_n_mt[key][TEST_SETTINGS['intp_model']]['intp_point_grad_mt']
        else: #If there is an error with interpolation
            print('Doing Simple Translation due to interpolation error')
            end_mt = gpt_translator.simple_translation(end['en'], target_lang=LANG_MAP[TEST_SETTINGS['lang']])
            test_cache[end['id']]['mts'].append(end_mt)
            grad_mts = []

        intp_n_mt[key][TEST_SETTINGS['intp_model']]["intp"] = intp_sents
        intp_n_mt[key][TEST_SETTINGS['intp_model']]["intp_point_grad_mt"] = grad_mts
        intp_n_mt[key][TEST_SETTINGS['intp_model']]["mt"] = end_mt

    #Make Final Output
    if(TEST_SETTINGS['aggregate_method'] == "none"):
        test_cache[end['id']]['mt'] = random.choice(test_cache[end['id']]['mts'])
    elif(TEST_SETTINGS['aggregate_method'] == "poll"):
        counter = Counter(test_cache[end['id']]['mts'])
        max_count = max(counter.values())
        max_elements = [key for key, count in counter.items() if count == max_count]
        test_cache[end['id']]['mt'] = random.choice(max_elements)
    elif(TEST_SETTINGS['aggregate_method'] == "prompt"):
        test_cache[end['id']]['mt'] = gpt_translator.aggr_translation(end['en'],test_cache[end['id']]['mts'], target_lang=LANG_MAP[TEST_SETTINGS['lang']])

test_result_filepath = f"your/file/path.json"

#Save Results
with open(test_result_filepath, 'w', encoding='utf-8') as f:
    json.dump(test_cache,f,ensure_ascii=False)

with open(f"data/intp_mt_{TEST_SETTINGS['lang']}.json",'w', encoding='utf-8') as f:
    json.dump(intp_n_mt, f, ensure_ascii=False)