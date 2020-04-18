import json
import re,os,sys
from pyjarowinkler import distance
import numpy as np
import nltk,pickle,tqdm
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

# sys.path.append('../util')
import utils
from utils import TextToVec,load_pickle,save_pickle,load_json, save_json
from utils import get_name_index



NEW_DATA_DIR = '../data/varible_test'          # original info, for test
NEW_DATA_V2_DIR = '../data/varible_train'
NEW_DATA_V3_DIR = '../data/varible'
WHOLE_AUTHOR_PROFILE_PATH = '../data/whole_author_profile.json'
WHOLE_AUTHOR_PROFILE_PUB_PATH = '../data/whole_author_profile_pub.json'


# WHOLE_AUTHOR_PROFILE_PATH = '../data/sample/whole_author_profile_sample.json'
# WHOLE_AUTHOR_PROFILE_PUB_PATH = '../data/sample/whole_author_profile_pub_sample.json'
SPLIT_DIR = '../data/split-data'


def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", "").replace("-", " ").replace("_", ' ').split()]
    full_name = '_'.join(x)
    return full_name

def convert_glove():
    print('convert glove begin !')
    word2vec = {}
    with open('../data/glove.840B.300d.txt', 'r') as f:
    # with open('D:\\kaggle\\Gendered Pronouns Resolution\\gap-master\\externals\\data\\glove.840B.300d.txt', 'r',encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            split_line = line.split()
            key = ' '.join(split_line[:-300])
            value = np.array(split_line[-300:], dtype=np.float)
            word2vec[key] = value
    print('Done')
    save_pickle(word2vec, '../data/glove.word2vec.dict.pkl')


def split_data(last_n_year):
    """
    split original data to train and test for paper-author wise feature.
    select the lastest paper as the test for every author.
    """
    def get_last_n_year_paper(n, paper_ids, whole_author_profile_pub):
        years = []
        for pid in paper_ids:
            year = whole_author_profile_pub[pid].get('year', '0')
            if year == '':
                year = 0
            else:
                year = int(year)
            if year < 1500 or year > 2100:
                year = 0
            years.append(year)
        # big to small
        years_sort_index = np.argsort(years)[::-1]
        target_index = years_sort_index[:n]
        paper_ids_array = np.array(paper_ids)
        return paper_ids_array[target_index]

    # assert last_n_year >= 2
    whole_author_profile = load_json(WHOLE_AUTHOR_PROFILE_PATH)
    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    test_profile = {}
    for aid in tqdm.tqdm(whole_author_profile):
        inner_dict = {}
        papers = whole_author_profile[aid]['papers']
        inner_dict['name'] = whole_author_profile[aid]['name']
        if len(papers) <= 1:
            inner_dict['papers'] = []
        elif len(papers) <= last_n_year:
            inner_dict['papers'] = get_last_n_year_paper(1, papers, whole_author_profile_pub).tolist()
        else:
            inner_dict['papers'] = get_last_n_year_paper(last_n_year, papers, whole_author_profile_pub).tolist()
        test_profile[aid] = inner_dict
        for pid in inner_dict['papers']:
            whole_author_profile[aid]['papers'].remove(pid)
        # years = []
        # for pid in papers:
        #     year = whole_author_profile_pub[pid].get('year', '0')
        #     if year == '':
        #         year = 0
        #     else:
        #         year = int(year)
        #     if year < 1500 or year > 2100:
        #         year = 0
        #     years.append(year)
        # last_year_index = np.argmax(years)
        # last_year_paper = papers[last_year_index]
        # whole_author_profile[aid]['papers'].remove(last_year_paper)
        # inner_dict['name'] = whole_author_profile[aid]['name']
        # inner_dict['papers'] = [last_year_paper]
        # test_profile[aid] = inner_dict
    os.makedirs(SPLIT_DIR, exist_ok=True)
    train_profile_path = os.path.join(SPLIT_DIR, 'train_profile-last%dyear.json' % last_n_year)
    test_profile_path = os.path.join(SPLIT_DIR, 'test_profile-last%dyear.json' % last_n_year)
    save_json(whole_author_profile, train_profile_path)
    save_json(test_profile, test_profile_path)


def sample_data(n_neg):
    """
    there are some 'bug' in the original sampled pair.
    try a more sensetive data sample method.
    we only choose the author's last year's paper as posi-pair, and other 2 authors as
    neg-pair.

    other posible reason is that it's a information leakage !!!!
    """
    def get_pid_with_index(whole_author_profile_pub, pid, name):
        authors = whole_author_profile_pub[pid]['authors']
        authors_names = [clean_name(item['name']) for item in authors]
        index = get_name_index(name, authors_names)
        return '%s-%d' % (pid, index)

    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    test_profile_path = os.path.join(SPLIT_DIR, 'test_profile-last1year.json')
    name2aids = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'name2aids.pkl'))
    test_profile = load_json(test_profile_path)
    # posi-pair, neg-pair --> [(aid, pid), ...]
    posi_pair = []
    neg_pair = []
    for aid in tqdm.tqdm(test_profile):
        name = test_profile[aid]['name']
        papers = test_profile[aid]['papers']
        papers = [get_pid_with_index(whole_author_profile_pub, pid, name) for pid in papers]
        if len(papers) == 0:
            continue
        # positive pair
        posi_pair.extend([(aid, pid) for pid in papers])

        # negative pair
        candidate_aids_ = name2aids[name]
        candidate_aids = candidate_aids_.copy().tolist()
        candidate_aids.remove(aid)
        if len(candidate_aids) == 0:
            continue
        candidate_aids = np.random.choice(candidate_aids, min([len(candidate_aids), n_neg]))
        neg_pair.extend([(neg_aid, pid) for neg_aid in candidate_aids for pid in papers])
    os.makedirs(NEW_DATA_V3_DIR, exist_ok=True)
    posi_pair_path = os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-extend%s.pkl' % n_neg)
    neg_pair_path = os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-extend%s.pkl' % n_neg)
    save_pickle(posi_pair, posi_pair_path)
    save_pickle(neg_pair, neg_pair_path)



def split_pair():
    pos_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'posi-pair-list-extend1.pkl'))
    neg_pair_set = load_pickle(os.path.join(NEW_DATA_V3_DIR, 'neg-pair-list-extend1.pkl'))
    pos_pid2aid = {}
    for pair in tqdm.tqdm(pos_pair_set):
        pos_pid2aid[pair[1]] = pair[0]
    neg_pid2aid = {}
    for pair in tqdm.tqdm(neg_pair_set):
        neg_pid2aid[pair[1]] = pair[0]
    pos_pid = set(pos_pid2aid.keys())
    neg_pid = set(neg_pid2aid.keys())
    innter_pid = list(pos_pid & neg_pid)
    # print('innter_pid:',len(innter_pid),'\n',innter_pid)

    train_pid, test_pid = train_test_split(innter_pid, test_size=0.15, shuffle=True)
    print(len(train_pid)) #14201
    print(len(test_pid)) #2507
    train_posi_pair_list = []
    train_neg_pair_list = []
    test_posi_pair_list = []
    test_neg_pair_list = []
    for pid in tqdm.tqdm(train_pid):
        train_posi_pair_list.append((pos_pid2aid[pid], pid))
        train_neg_pair_list.append((neg_pid2aid[pid], pid))
    for pid in tqdm.tqdm(test_pid):
        test_posi_pair_list.append((pos_pid2aid[pid], pid))
        test_neg_pair_list.append((neg_pid2aid[pid], pid))
    save_pickle(train_posi_pair_list, os.path.join(NEW_DATA_V3_DIR, 'train-posi-pair-list.pkl'))
    save_pickle(train_neg_pair_list, os.path.join(NEW_DATA_V3_DIR, 'train-neg-pair-list.pkl'))
    save_pickle(test_posi_pair_list, os.path.join(NEW_DATA_V3_DIR, 'test-posi-pair-list.pkl'))
    save_pickle(test_neg_pair_list, os.path.join(NEW_DATA_V3_DIR, 'test-neg-pair-list.pkl'))

def get_triplet_corpus(mission='train'):
    whole_author_profile_pub = load_json(WHOLE_AUTHOR_PROFILE_PUB_PATH)
    whole_author_profile = load_json(WHOLE_AUTHOR_PROFILE_PATH)
    name2aids = {}
    aid2pids = {}
    aids = []
    names = []
    pids_with_index = []
    for aid in tqdm.tqdm(whole_author_profile):
        aids.append(aid)
        names.append(whole_author_profile[aid]['name'])
        pids = whole_author_profile[aid]['papers']
        tmp = []
        for paper in pids:
            paper_authors = whole_author_profile_pub[paper]['authors']
            author_names = [clean_name(item['name']) for item in paper_authors]
            # print(author_names)
            index = get_name_index(names[-1], author_names)
            tmp.append('%s-%d' % (paper, index))
        pids_with_index.append(tmp)
    assert len(aids) == len(names)
    assert len(names) == len(pids_with_index)
    print('all aids num: ', len(aids))
    name_set = set(names)
    names_array = np.array(names)
    aids_array = np.array(aids)
    for name in name_set:
        target_aid = aids_array[names_array == name]
        name2aids[name] = target_aid
    for aid, pid in zip(aids, pids_with_index):
        aid2pids[aid] = pid
    if mission == 'train':
        save_pickle(name2aids, os.path.join(NEW_DATA_V2_DIR, 'name2aids.pkl'))
        save_pickle(aid2pids, os.path.join(NEW_DATA_V2_DIR, 'aid2pids.pkl'))
    elif mission == 'test':
        save_pickle(name2aids, os.path.join(NEW_DATA_DIR, 'name2aids.pkl'))
        save_pickle(aid2pids, os.path.join(NEW_DATA_DIR, 'aid2pids.pkl'))


    texttovec = TextToVec()
    if mission == 'train':
        aid2pids = load_pickle(os.path.join(NEW_DATA_V2_DIR, 'aid2pids.pkl'))
    elif mission == 'test':
        aid2pids = load_pickle(os.path.join(NEW_DATA_DIR, 'aid2pids.pkl'))
    # ------------------------------------------
    # save format: aid2titlevec --> {aid: [mean value]}
    aid2titlevec = {}
    for aid in tqdm.tqdm(aid2pids.keys()):
        papers = aid2pids[aid]
        inner_list = []
        for pid_with_index in papers:
            pid, index = pid_with_index.split('-')
            title = whole_author_profile_pub[pid]['title']
            inner_list.append(texttovec.get_vec(title))
        if len(inner_list) == 0:
            aid2titlevec[aid] = np.zeros(300)
        else:
            aid2titlevec[aid] = np.mean(np.array(inner_list), axis=0)
    if mission == 'train':
        save_pickle(aid2titlevec, os.path.join(NEW_DATA_V2_DIR, 'aid2titlevec.pkl'))
    elif mission == 'test':
        save_pickle(aid2titlevec, os.path.join(NEW_DATA_DIR, 'aid2titlevec.pkl'))


def gen_bert_corpus(fn_author='../data/train_author.json', fn_paper='../data/train_pub.json',all_fn_author='../data/whole_author_profile.json',
                   all_fn_paper='../data/whole_author_profile_pub.json',fn_train_instances='../data/instance/train_instance_new.pkl'):


    train_instances = pickle.load(open(fn_train_instances, "rb"))
    # print("train_instances:",train_instances)
    #
    # train_instances = [({("cHiKoqn9", '', "Wtx5Pe3T")}, {})]

    with open(fn_author, "r") as fa, open(fn_paper, "r") as fp , open(all_fn_author, "r") as all_fa, open(all_fn_paper,
                                                                                                         "r") as all_fp:

        author_json = json.load(fa)
        paper_json = json.load(fp)

        all_author_json = json.load(all_fa)
        all_paper_json = json.load(all_fp)

        # all_paper_json = paper_json

        corpus_list = []
        for instance in train_instances:
            pos_ins_set, neg_ins_set = instance
            # print("pos_ins_set:",pos_ins_set)
            # print("neg_ins_set:",neg_ins_set)

            for pos_ins in pos_ins_set:
                paper_id, paper_org, paper_author_id = pos_ins

                paper_info = all_paper_json[paper_id]
                title = ""

                if "title" in paper_info and (paper_info["title"] != None) and (len(paper_info["title"]) > 0):
                    title = paper_info["title"]

                # print(title)

                paperIds = all_author_json[paper_author_id]["papers"]
                # print(paperIds)


                now_corpus = ""




                for pid in paperIds:
                    if pid != paper_id:
                        now_paper = all_paper_json[pid]

                        if "title" in now_paper and (now_paper["title"] != None) and (len(now_paper["title"]) > 0):

                            now_corpus += now_paper["title"]

                if title != "" and now_corpus != "":
                    row = [title,now_corpus,  1]
                    corpus_list.append(row)









            for neg_ins in neg_ins_set:
                paper_id, paper_org, paper_author_id = neg_ins
                paper_info = all_paper_json[paper_id]
                title = ""

                if "title" in paper_info and (paper_info["title"] != None) and (len(paper_info["title"]) > 0):
                    title = paper_info["title"]

                paperIds = all_author_json[paper_author_id]["papers"]

                now_corpus = ""

                for pid in paperIds:
                    if pid != paper_id:
                        now_paper = all_paper_json[pid]

                        if "title" in now_paper and (now_paper["title"] != None) and (
                                len(now_paper["title"]) > 0):
                            now_corpus += now_paper["title"]

                if title != "" and now_corpus != "":
                    row = [now_corpus, title, 0]
                    corpus_list.append(row)

        df = pd.DataFrame(corpus_list,
                          columns=['paper_abs', 'author_corpus', 'label'])

        df.to_csv("../data/bert_corpus.csv")

if __name__ == "__main__":
    # gen_bert_corpus(
    #
    #     fn_train_instances='../data/instance/train_instance.pkl')
    # convert_glove()
    split_data(1)
    get_triplet_corpus('train')
    get_triplet_corpus('test')
    # for i in [1, 2, 3]:
    sample_data(1)
    split_pair()

