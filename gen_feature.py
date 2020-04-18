import os
import pickle
from collections import defaultdict
from pyjarowinkler import distance
import jellyfish
import json
import random
import numpy as np
import pandas as pd
import difflib, Levenshtein
from tqdm import tqdm
# # from baseline import get_author_info
import sys
sys.path.append("../util")
import nltk
# from baseline import get_author_info
from data_process import aligh_author_name
from data_process import get_converse_name
from data_process import char_filter
from  doc_similarity import DOC_SIM
from pyjarowinkler import distance
from data_process import get_author_set
from data_process import get_abbrevname_match_org
from data_process import get_substr_org
from data_process import is_same_abbrev_name
from nn_bert_pointwise import convert_data
from triplet_model import TripletModel
import torch

bert_model = torch.load('../model/model_bert_final.pkl_1')

bert_model.cuda()

bert_model.eval()

class DataCollection:
    def __init__(self ,fn_author_info="../data/all_author_info.txt",
                 fn_author_info_year="../data/all_author_info_with_year.pickle",

                 fn_train_pub='../data/train_pub.json',
                 fn_train_author="../data/train_author.json",
                 fn_whole_author='../data/whole_author_profile.json',
                 fn_whole_pub = '../data/whole_author_profile_pub.json',
                 fn_valid_pub="../data/cna_valid_pub.json"):

        self.paper_pub_train = {}
        self.author_train = {}
        self.author_whole = {}
        self.paper_pub_cna = {}
        self.paper_pub_whole = {}
        with open(fn_train_pub, "r") as papers, open(fn_train_author, "r") as authors, open(
                fn_valid_pub, "r") as fn_paper_pub_cna, open(fn_whole_author ,"r") as whole_author, \
                open(fn_whole_pub ,"r") as  whole_pub:
            self.paper_pub_train = json.load(papers)
            self.author_train = json.load(authors)
            self.paper_pub_cna = json.load(fn_paper_pub_cna)
            self.author_whole = json.load(whole_author)
            self.paper_pub_whole = json.load(whole_pub)

        self.split_chars = ['\u202f']  # '\u2212'=='-'
        self.stoplist = nltk.corpus.stopwords.words('english')

        self.whole_name_list = ['li_guo', 'bo_shen', 'di_wang', 'long_wang', 'qiang_xu', 'xiang_wang', 'changming_liu', 'kenji_kaneko', 'guohua_chen', 'hai_jin', 'jia_li', 'guoliang_li', 'lan_wang', 'alessandro_giuliani', 'jiang_he', 'xiang_gao', 'jianping_wu', 'peng_shi', 'feng_wu', 'jing_zhu', 'xiaoyang_zhang', 'mei_han', 'jianhua_shen', 'hong_xiao', 'qing_yang', 'makoto_sasaki', 'feng_gao', 'd_zhang', 'akira_ono', 'hong_yang', 'bin_gao', 't_wang', 'qing_liu', 'ping_sheng', 'j_huang', 'chao_yuan', 'jianyong_wang', 'hong_yan_wang', 'xi_liu', 'jing_chen', 'bo_zhou', 'bin_yao', 'ping_yang', 'jianchao_yang', 'f_liu', 'minhua_jiang', 'j_lin', 'hong_li', 'ke_li', 'jing_deng', 'xin_li', 'xin_wu', 'alexander_belyaev', 'chen_liu', 'di_wu', 'xiao_liu', 'bin_zhao', 'qian_chen', 'chao_zhang', 'w_huang', 'ning_li', 'kai_sun', 'feng_zhao', 'hao_huang', 'israel_goldberg', 'hui_wei', 'yajun_guo', 'jianfeng_gao', 'xue_wang', 'lei_zhao', 'meng_wang', 'yuanyuan_liu', 'dongsheng_wang', 'qian_zhang', 'koji_fujita', 'changhong_wang', 'heqing_huang', 'weiguo_fan', 'jianping_fan', 'yan_liang', 'l_sun', 'liang_wang', 's_huang', 'jinlong_yang', 'jian_wang', 'tao_jiang', 'jing_gao', 'feng_li', 'lin_zhong', 'fang_wang', 'liang_li', 'c_c_lin', 'wen_tan', 'xing_liu', 'feng_zhu', 'jian_zhou', 'shuai_zhang', 'juan_liu', 'li_jiang', 'yong_yu', 'bing_chen', 'liang_han', 'jin_zhang', 'tao_zhou', 'jie_zhang', 'xiaoping_wu', 'jianguo_liu', 'liping_liu', 'song_gao', 'di_zhang', 'yue_wang', 'jianmin_zhao', 'jianjun_liu', 'rakesh_kumar', 'yu_gu', 'jin_luo', 'shaobin_wang', 'jin_he', 'hui_yu', 'lei_chen', 'wenjie_shen', 'fei_qi', 'xiaoming_liu', 'shu_tao', 'j_ma', 'weidong_li', 'd_li', 'haiying_wang', 'dan_wu', 'akio_kobayashi', 'baoli_zhu', 'li_ma', 'y_feng', 'z_zhou', 'huan_liu', 'furong_gao', 'chuan_he', 'lin_xu', 'jianxin_shi', 'hao_yan', 'liu_yang', 'j_jiang', 'liang_liu', 'hong_liu', 'kui_ren', 'jing_wu', 'dong_xu', 'huan_yang', 'lihua_xie', 'guochun_zhao', 'li_zhou', 'jie_lin', 'guanghua_li', 'feng_liu', 'hong_mei', 'xiaodong_he', 'hong_cheng', 'zhi_gang_zhang', 'c_c_wang', 'lei_jiang', 'shan_gao', 'chenguang_wang', 'jie_xiao', 'li_li_wang', 'bo_zhang', 'bin_zhang', 'ji_zhang', 'cheng_yang', 'weiping_cai', 'gang_pan', 'xi_li', 'guoqing_hu', 'jianguo_hou', 'haijiang_wang', 'bing_zhao', 'wei_chen', 'jianfeng_chen', 'wei_zheng', 'juan_chen', 'guang_yang', 'hong_zhao', 'chao_deng', 'fei_liu', 'cheng_zhu', 'min_sun', 'hui_wang', 'chao_chen', 'gang_zhang', 'xiaoming_li', 'chunxia_li', 'wenjun_zheng', 'bo_jiang', 'hideaki_takahashi', 'kai_yu', 'changsheng_li', 'peng_gong', 'hongbo_sun', 'jie_yin', 'z_wu', 'zhi_wang', 'lixia_zhang', 'qiong_luo', 's_liu', 'jin_li', 'jing_li', 'c_yang', 'yuxi_liu', 'bing_zhang', 'j_y_wang', 'sheng_dai', 'jie_luo', 'bin_hu', 'hai_yang', 'ming_chen', 'juan_du', 'hiroaki_okamoto', 'dan_wang', 'jun_chen', 'bai_yang', 'hiroshi_sakamoto', 'jie_yang', 'li_he', 'fei_huang', 'hang_li', 'hui_zhang', 'jian_chen', 'bin_liu', 'lei_zhu', 'm_yang', 'tong_zhang', 'j_conway', 'jin_xie', 'fusheng_wang', 'jie_liu', 'jun_yang', 'yan_gao', 'xia_zhang', 'yi_jiang', 'liang_cheng', 'xia_li', 'guoqiang_chen', 'osamu_watanabe', 'xi_zhang', 'yi_chen', 'guoping_zhang', 'bin_jiang', 'chen_chen', 'jie_tian', 'm_li', 'jing_jin', 'hong_he', 'bo_li', 'jing_yu', 'lei_shi', 'jin_xu', 'kai_li', 'hua_li', 'li_deng', 'bo_tang', 'ning_zhou', 'bin_han', 'qing_zhang', 'hui_gao', 'bin_yu', 'wenjun_zhang', 'lin_he', 'bing_liu', 'l_zhao', 'kun_zhou', 'wei_song', 'chen_dong', 'qiwei_zhang', 'dan_xie', 'jianping_ding', 'rui_zhang', 'zhe_zhang', 'jing_tian', 'c_h_chen', 'jianjun_yu', 'kai_chen', 'bing_zhou', 'bo_xu', 'kun_zhang', 'li_gao', 'min_chen', 'juan_zhang', 'ming_jiang', 'hideyuki_suzuki', 'huijun_zhao', 'sheng_xu', 'feng_ding', 'lu_liu', 'ling_liu', 'chunming_xu', 'jie_xu', 'jihyun_kim', 'qin_xin', 'dong_yu', 'a_mukherjee', 'hui_xiong', 'hong_fang', 'li_mao', 'tao_xie', 'cong_yu', 'jinghong_li', 'hiroshi_nakanishi', 'jin_liu', 'hongzhi_wang', 'jie_zhu', 'he_tian', 'hongxing_xu', 'can_li', 'hongyu_zhao', 'kun_liu', 'qingsong_xu', 'jincai_zhao', 'weihong_zhu', 'wenping_hu', 'jianfang_wang', 'fei_wei']

        self.train_instances = []
        self.paper2aid2name = {}  # paper2aid2name.setdefault(paper_id,[]).append ((author_name, author_id))

        self.paper_authors = dict()  # {paperId: authorIds} data from train_author.json , whole_author.json
        self.author_papers = dict()  # {authorId:paperIds} data from train_author.json , whole_author.json

        # self.authorconference[author][conferenceId] += 1
        self.author_venue = dict()  # {authorId: {venueStr：num}}

        self.paper_venue = dict()  # data from Paper_pub

        self.paper_year = dict()
        self.paper_keywords = dict()
        self.paper_title = dict()
        self.paper_abstract = dict()

        self.paper_all_keywords = dict()

        self.author_info = dict() # {authorId:{name:[org_list] }}
        # self.author_info_year = pickle.load(open(fn_author_info_year,"rb"))
        self.author_name_ids = dict()  # {name:[authorIds]}
        self.author_year_avg = dict()
        self.author_paper_num = dict()

        self.author_sim_model = dict()

        self.avgaffiliationlength = 0.0
        self.avgauthornamelength = 0.0
        self.avg_title_length = 0.0  # 95.59173179989008


        self.year_range = dict()
        self.year_range_score = dict()  # {0: 0.4294701092200224, 19: 0.0025471807340511753, 18: 0.0028173362664505423, 7: 0.03546756203928833, 5: 0.0426845741191, 12: 0.008336227856894755, 2: 0.09389834433252287, 10: 0.013391995677511481, 3: 0.0742155841148547, 8: 0.024313997915943037, 9: 0.01813901431824322, 1: 0.11898421519817838, 6: 0.037898961830882635, 4: 0.05260314152290533, 14: 0.006059202655242947, 23: 0.0007718729497124773, 11: 0.009764192813862838, 17: 0.0039365520435336344, 39: 0.0001157809424568716, 21: 0.0012349967195399638, 13: 0.009146694454092856, 16: 0.0044768631083323684, 15: 0.004438269460846744, 25: 0.000578904712284358, 22: 0.0010806221295974682, 30: 7.718729497124774e-05, 20: 0.0013121840145112116, 29: 0.0002315618849137432, 38: 7.718729497124774e-05, 24: 0.0006174983597699819, 31: 0.0001157809424568716, 40: 3.859364748562387e-05, 27: 0.00038593647485623867, 26: 0.00027015553239936706, 28: 0.0002315618849137432, 49: 7.718729497124774e-05, 37: 3.859364748562387e-05, 35: 3.859364748562387e-05, 36: 3.859364748562387e-05, 32: 3.859364748562387e-05, 42: 3.859364748562387e-05}


        self.wrong_author_name_list = ['jianguo_liu', 'bo_shen', 'qiwei_zhang', 'fusheng_wang', 'hui_wang','hong_xiao', 'jin_zhang','bin_yao','jing_gao',
                                  'jing_chen','hong_cheng','qiang_xu','feng_liu']

        ################################################
        self.author_info = self.get_authorId_name_orgs(fn_author_info) #
        self.author_info_year = pickle.load(open(fn_author_info_year, "rb"))
        self.author_name_ids = self.get_author_info_name_key(fn_author_info) # author_info_name_key.setdefault(name, []).append(id)
        self.get_paper_authors()
        self.get_paper_pub() # get year title keywords
        # print(self.paper_year["7BasHa4Q"])

        self.get_year_range()
        # self.get_author_sim_model()

    def unicode_filter(self,str):
        str = str.replace('\u202f',' ')
        str = str.replace('\u2212', '−')
        str = str.replace('\u2013', '-') #–
        str = str.replace('\u2218', '.')#∘
        # str = str.replace('\u03bc', ' ')#μ $$$$ \u03b3

        str = str.replace('\u2019','’')
        str = str.replace('\u00a0',' ')
        str = str.replace('\u00b0','°')
        str = str.replace('&#x',' ')
        str = str.replace( " and ",' & ')
        str = str.replace('&amp;', '&')
        str = str.replace('%%#',' ')
        str = str.replace('~',' ')
        str = str.replace('$$$$',' ')
        str = str.replace('^',' ')
        str = str.replace('&quot;','"')


        '''
        %%#%%#~*'''

        return str



    def jaccard(self,set1, set2):
        union = len(set1 | set2)
        intersect = len(set1 & set2)
        if union > 0:
            return float(intersect) / float(union)
        else:
            return -1

    def convert_data(data, max_seq_length_a=200, max_seq_length_b=500, tokenizer=None):
        all_tokens = []
        longer = 0
        # for row in data.itertuples():
        tokens_a = tokenizer.tokenize(data[0])
        tokens_b = tokenizer.tokenize(data[1])
        if len(tokens_a) > max_seq_length_a:
            tokens_a = tokens_a[:max_seq_length_a]
            longer += 1
        if len(tokens_a) < max_seq_length_a:
            tokens_a = tokens_a + [0] * (max_seq_length_a - len(tokens_a))
        if len(tokens_b) > max_seq_length_b:
            tokens_b = tokens_b[:max_seq_length_b]
            longer += 1
        if len(tokens_b) < max_seq_length_b:
            tokens_b = tokens_b + [0] * (max_seq_length_b - len(tokens_b))
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"])
        all_tokens.append(one_token)
        # data['bert_token'] = all_tokens
        return all_tokens

    #useless
    def get_year_range(self):
        ind = 0
        for authorId, paperIds in self.author_papers.items():
            # if ind > 5 :
            #     break
            year_min = 3000
            pmax = 0
            pmin = 0
            year_max = -1
            for pid in paperIds:


                if (pid in self.paper_year) and   self.paper_year[pid ]!= "" and self.paper_year[pid] != 0:
                    p_year = self.paper_year[pid]
                    if p_year > year_max:
                        year_max = p_year
                        pmix = pid
                    if p_year < year_min:
                        year_min = p_year
                        pmin = pid
            yrange = year_max - year_min

            # if yrange == 303:
            #     print(authorId,year_max,year_min, pmax, pmin)


            self.year_range.setdefault(yrange ,0)
            if yrange not in self.year_range:
                self.year_range.setdefault(yrange, 0)
            else:
                self.year_range[yrange] = self.year_range[yrange] + 1

            # print( yrange)


            # ind += 1
        # print('len self.author_papers:',len(self.author_papers)) # 25911
        for yr, cnt in self.year_range.items():
            # print(yr,' ',cnt)
            self.year_range_score[yr] = float(cnt ) /len(self.author_papers)
        self.year_range_score["unknown"] = 0

        # print(self.year_range_score)




    def get_paper_authors(self):




        for (key, value) in self.author_train.items():
            author_name = key  # li_guo
            # print(author_name)
            # print(value)
            for authorId ,paperIds in value.items():
                # author_info.setdefault((authorId, author_name), set())
                # print(authorId)
                for paperId in paperIds:
                    self.paper_authors.setdefault(paperId ,set()).add(authorId)
                    self.author_papers.setdefault(authorId ,set()).add(paperId)

        for (key, value) in self.author_whole.items():
            authorId = key  # li_guo
            info = value

            # author_name = info["name"]
            # author_info.setdefault((authorId, author_name), set())
            paperIds = info["papers"]
            for paperId in paperIds:
                self.paper_authors.setdefault(paperId, set()).add(authorId)
                self.author_papers.setdefault(authorId, set()).add(paperId)

        for authorId, paperIds in self.author_papers.items():
            self.author_paper_num[authorId] = len(paperIds)



    #useless
    def get_author_sim_model(self): #要在get_paper_pub之后执行
        for (key, value) in self.author_train.items():
            author_name = key #li_guo
            # print(author_name)
            # print(value)
            for authorId,paperIds in value.items():
                # author_info.setdefault((authorId, author_name), set())
                # print(authorId)
                documents = []
                paperIds = list(set(paperIds))
                for paperId in paperIds:
                    if paperId in self.paper_abstract:
                        documents.append(self.paper_abstract[paperId])

                if len(documents) > 0:
                    documents = list( set( documents))
                    if len(documents) > 1:

                        try:
                            self.author_sim_model[authorId] = DOC_SIM(documents)
                        except ValueError:
                            pass



        for (key, value) in self.author_whole.items():
            authorId = key #li_guo
            info = value

            # author_name = info["name"]
            # author_info.setdefault((authorId, author_name), set())
            if authorId not in self.author_sim_model:

                paperIds = info["papers"]
                documents = []
                paperIds = list(set(paperIds))
                for paperId in paperIds:
                    if paperId in self.paper_abstract:
                        documents.append(self.paper_abstract[paperId])

                if len(documents) > 0:
                    documents = list( set( documents))
                    if len(documents) > 1:
                        try:
                            self.author_sim_model[authorId] = DOC_SIM(documents)
                        except ValueError:
                            pass


    def get_paper_pub(self):  # get year , title, keywords

        pubs = [self.paper_pub_whole, self.paper_pub_train, self.paper_pub_cna]

        distinct_title_count = 0  # 271111
        for pub in pubs:
            for paperId, value in pub.items():

                if "keywords" in value and (len(value["keywords"]) > 0):  # ""  []
                    #

                    if paperId not in self.paper_keywords:
                        kws = value["keywords"]
                        kws_str = " ".join([kw  for  kw in kws if kw != "" and kw != "null"]).lower()

                        kws_list = [w for w in
                                        kws_str.replace(",", ' ').replace(".", ' ').replace("|", ' ').replace("-", ' ')
                                       .replace(";",' ').replace(":", ' ').replace("(", ' ').replace(")", ' ').split() if not w in self.stoplist] #:

                        # str = str.strip()
                        if len(kws_list) > 0:
                            self.paper_keywords[paperId] = list( set(kws_list) )


                            self.paper_all_keywords[paperId] = set( list( set(kws_list) ) )




                if "title" in value and (len(value["title"]) > 0):
                    if paperId not in self.paper_title:
                        title = value["title"]
                        fil_title = char_filter(title)
                        if fil_title != "":
                            self.paper_title[paperId] = fil_title
                            distinct_title_count += 1
                            self.avg_title_length += float(len(title))

                            self.paper_all_keywords.setdefault(paperId ,set()).update(fil_title.split())
                        # elif self.paper_title[paperId] != value["title"].lower():
                        #     print("title error!:" ,paperId)
                        #     exit()



                if "venue" in value and (len(value["venue"]) > 0):
                    if paperId not in self.paper_venue:
                        v = value["venue"]
                        self.paper_venue[paperId] = v.lower()
                    elif self.paper_venue[paperId] != value["venue"].lower():
                        print("venue error!:" ,paperId)
                        exit()

                if "abstract" in value and  (value["abstract"] != None) and (len(value["abstract"]) > 0):
                    if paperId not in self.paper_abstract:

                        v = char_filter(value["abstract"])
                        self.paper_abstract[paperId] = v
                        self.paper_all_keywords.setdefault(paperId, set()).update(v.split())



        self.avg_title_length /= float(distinct_title_count) # 95.59173179989008


        for paperId, value in self.paper_pub_whole.items():
            # print(paperId)
            if "year" in value:
                if paperId not in self.paper_year :
                    if value["year"] != "" and value["year"] != 0 and value["year"] > 1000 and value["year"] < 2021: #排除了210
                        self.paper_year[paperId] = value["year"]
                # elif self.paper_year[paperId] != value["year"] : 没有出现此状况
                #     print("error")
                #     print(paperId)
                #     exit()
        #     if "title" in value:
        #         title = value["title"]
        #         self.paper_title[paperId] = title.lower()
        #         distinct_title_count += 1
        #         self.avg_title_length += float(len(title))
        #
        #     if "keywords" in value:
        #         keywords = value["keywords"] #list
        #         keywords = [w for w in keywords.lower().replace('"', ' ').replace("|", ' ').replace(",", ' ').split() if
        #                 not w in stoplist]
        #
        #
        # self.avg_title_length /= float(distinct_title_count)
        # print('self.avg_title_length:',self.avg_title_length)


        for paperId, value in self.paper_pub_train.items():
            if "year" in value:
                if paperId not in self.paper_year :
                    if value["year"] != "" and value["year"] != 0 and value["year"] > 1000 and value["year"] < 2021:
                        self.paper_year[paperId] = value["year"]

        for paperId, value in self.paper_pub_cna.items():
            if "year" in value:
                if paperId not in self.paper_year:
                    if value["year"] != "" and value["year"] != 0 and value["year"] > 1000 and value["year"] < 2021:
                        self.paper_year[paperId] = value["year"]


        author_pubs = [self.author_whole, self.author_train]

        # get author_year_avg ###################################################

        for (key, value) in self.author_whole.items():
            authorId = key  # li_guo
            info = value

            paperIds = info["papers"]
            year_list = []
            for paperId in paperIds:
                if paperId in self.paper_year:
                    year_list.append(self.paper_year[paperId])

            self.author_year_avg[authorId] = np.average(year_list)

        # get author - venue ###################################################################
        for (key, value) in self.author_train.items():
            author_name = key  # li_guo

            for authorId ,paperIds in value.items():

                for paperId in paperIds:
                    if paperId in self.paper_venue:
                        self.author_venue.setdefault(authorId ,{}).setdefault(self.paper_venue[paperId] ,0)
                        self.author_venue[authorId][self.paper_venue[paperId]] = self.author_venue[authorId][self.paper_venue[paperId]] +1

        for (key, value) in self.author_whole.items():
            authorId = key  # li_guo
            info = value


            paperIds = info["papers"]
            for paperId in paperIds:
                if paperId in self.paper_venue:
                    if self.author_venue.setdefault(authorId, {}).setdefault(self.paper_venue[paperId], 0) == 0:
                        self.author_venue[authorId][self.paper_venue[paperId]] = self.author_venue[authorId][
                                                                                     self.paper_venue[paperId]] + 1

        # print("iOWnMOe9:",self.author_venue["iOWnMOe9"]) #iOWnMOe9: {'journal of microencapsulation': 1, 'current drug delivery': 2, 'current nanoscience': 1}

    def get_authorId_name_orgs(self, fn_all_author_info):
        author_info = {}
        se = set()
        with open(fn_all_author_info, "r", encoding='utf-8') as all_author_info:
            for line in all_author_info:

                if len(line.strip().split('\t')) == 3:
                    # print(line.split())
                    id, name, org_list = line.strip().split('\t')

                    org_list = org_list.split("|||")
                    org_list = list(map(lambda x: x.lower(), org_list))
                    author_info.setdefault(id, {}).setdefault(name, []).extend(org_list)
                elif len(line.strip().split('\t')) == 2:
                    # print(line.split())
                    id, name = line.strip().split('\t')

                    author_info.setdefault(id, {}).setdefault(name, [])
        # print('author_info:', author_info['CIJI5QBY'])

        return author_info

    def get_author_info_name_key(self, fn_all_author_info):  # name ids pair
        author_info_name_key = {}
        with open(fn_all_author_info, "r", encoding='utf-8') as all_author_info:
            for line in all_author_info:
                id, name = line.strip().split('\t')[:2]

                author_info_name_key.setdefault(name, []).append(id)

        return author_info_name_key

    def get_match_org(self, paperId, authorId, paper_pub):

        author_name = list(self.author_info[authorId].keys())[0]
        # print(author_name)

        paper_authors = paper_pub[paperId]["authors"]
        match_flag = False
        for a_info in paper_authors:
            name = a_info["name"]
            org = a_info["org"] if "org" in a_info else ""

            if aligh_author_name(name) == author_name:

                return org
        print("instance none match:",paperId, authorId, author_name)
        return ""


    def get_match_org0(self, paperId, authorId, paper_pub):

        paper = paper_pub[paperId]
        author_list = paper["authors"]
        author_list = get_author_set(author_list)  # 去重
        author_name = list(self.author_info[authorId].keys())[0]

        match_num = 0

        target_org = None

        # print(author_name,paperId)
        for o in author_list:  # Li Guo
            if o["name"].find(" ") == -1 and o["name"].find("\u00a0") == -1:  # Lin-Qi Jianmin\u00a0Zhao == jianmin_zhao
                continue
            if aligh_author_name(o["name"]) == author_name or get_converse_name(o["name"]) == author_name:
                match_num = match_num + 1
                # print("aligh_author_name:",aligh_author_name( o["name"] ),get_converse_name( o["name"] ))
                if "org" in o:
                    org = o["org"]
                    if org != "":

                        # match_num = match_num + 1
                        org = org.replace("\n", '\\n').replace("\t", " ")
                        org = char_filter(org)  # lower #"." -> 导致空
                        # global author_info
                        if org != "":
                            target_org = org
        if match_num > 1:
            target_org = None

            print("match overflow:", author_name, paperId)
        elif match_num == 0:

            abbrev_match_org, abbrev_match_flag = get_abbrevname_match_org(author_name, author_list)
            if abbrev_match_org != None:
                fil_org = char_filter(abbrev_match_org)
                if fil_org != "":
                    target_org = fil_org
                    print("from abbreve")

            else:
                sub_match_org, sub_match_flag = get_substr_org(author_name, author_list)
                if sub_match_org != None:

                    fil_org = char_filter(sub_match_org)
                    if fil_org != "":
                        target_org = fil_org
                        print("from sub name")
                else:
                    if abbrev_match_flag == False and sub_match_flag == False:
                        print("none match :", author_name, paperId)




        else: #match_num == 1:
            # if target_org == None:
            #     print("error! match org exception!")
            pass
        if target_org == None:
            target_org = ""
        return target_org  # 并集


    def clean_name(self, name):
        if name is None:
            return ""
        x = [k.strip() for k in name.lower().strip().replace(".", "").replace("-", " ").replace("_", ' ').split()]
        # x = [k.strip() for k in name.lower().strip().replace("-", "").replace("_", ' ').split()]
        full_name = ' '.join(x)
        name_part = full_name.split()
        if (len(name_part) >= 1):
            return full_name
        else:
            return None

    def get_author_index(self, author_name, author_list):
        # instance: ('vkUqD3U8', '', '6tBpMRDL')
        # author_list: ['dun_wentao', 'jia_shuheng', 'qiu_xiurong', 'yuan_chao', 'bi_qingsheng', 'cai_bin', 'li_mian']
        # ----ValueError: 'chao_yuan' is not in list
        # return author_list.index(author_name)
        # 找出paper中author_name所对应的位置

        score_list = []
        name = self.clean_name(author_name)
        author_list_lower = []
        for author in author_list:
            author_list_lower.append(author.lower())
        name_split = name.split()
        for author in author_list_lower:
            # lower_name = author.lower()
            score = distance.get_jaro_distance(name, author, winkler=True, scaling=0.1)
            author_split = author.split()
            inter = set(name_split) & set(author_split)
            alls = set(name_split) | set(author_split)
            score += round(len(inter) / len(alls), 6)
            score_list.append(score)

        rank = np.argsort(-np.array(score_list))
        return_list = [author_list_lower[i] for i in rank[1:]]

        return rank[0]


    def load_train_instances(self, fn='../data/instance/train_instance.pkl'):
        self.train_instances = pickle.load(open(fn, "rb"))
        # print("train_instances_load:", self.train_instances)
        print("训练样本大小（load）：",len(self.train_instances))


    # gen_train_instances('../data/train_author.json', '../data/train_pub.json', 10000)
    def gen_train_instances(self, instance_num, neg_sample_num, fout='../data/instance/train_instance.pkl'):

        author_data = self.author_train
        paper_data = self.paper_pub_train

        name_train = set()

                # 筛选训练集，只取同名作者数大于等于5个的名字作为训练集。
        for name in author_data:
            if name in self.wrong_author_name_list:
                continue
            persons = author_data[name]
            if (len(persons) > 5):
                name_train.add((name))

        print('len(name_train):',len(name_train))  # 196 -> 185
        # cnt = 0
        for author_name in name_train:  # author_data name
            persons = author_data[author_name]
            for author_id in persons:  # author_id
                paper_list = persons[author_id]
                for paper_id in paper_list:
                    self.paper2aid2name.setdefault(paper_id, []).append(
                        (author_name, author_id))  # from author.json 保存name是为了负采样

        print('len(self.paper2aid2name):',len(self.paper2aid2name))  # 198607 -> 174606
        # print(cnt) #2143

        total_paper_list = list(self.paper2aid2name.keys())  # total_paper_list
        # print("训练样本论文总个数:",len(total_paper_list))

        # 采样10000篇paper作为训练集
        # train_paper_list = total_paper_list
        train_paper_list = random.sample(total_paper_list, instance_num)



        for paper_id in train_paper_list:
            # print("paperId:",paper_id)

            # 保存对应的正负例
            pos_ins = set()
            neg_ins = set()

            pos_num = 0
            now_author_name = -1
            # now_author_id = -1 #可能有两个同名作者写了同一篇论文 bo shen
            tot_author_ids = []
            # if len(paper2aid2name[paper_id]) > 1:
            #     # print("有多个作者写了此篇论文:", paper_id, paper2aid2name[paper_id])
            index = 0
            for paper_author_name, paper_author_id in self.paper2aid2name[paper_id]:


                paper_org = self.get_match_org0(paper_id, paper_author_id, paper_pub=self.paper_pub_train)
                # print("match org:",paper_id, paper_author_id,paper_author_name)
                # print("match org:",paper_org)
                if paper_org == None:
                    paper_org = ""
                # print("match org:", paper_id, paper_author_id, paper_author_name,paper_org)
                pos_ins.add((paper_id, paper_org, paper_author_id))
                pos_num = pos_num + 1

                persons = list(author_data[paper_author_name].keys())  # ../data/train_author.json
                # print("now_author_name:",now_author_name)

                persons.remove(paper_author_id)
                assert len(persons) == (len(list(author_data[paper_author_name].keys())) - 1)

                ''' 
                #按发表paper数取top
                
                tmp_persons = []
                for per in persons:
                    tmp_persons.append((per, self.author_paper_num[per]))

                tmp_persons = sorted(tmp_persons, key=lambda x: x[-1], reverse=True)
                persons_top5 = tmp_persons[:5] #persons_top5: [('xCwVdbP2', 36), ('UD9Dl2N0', 17), ('LAy5I3sn', 14), ('cwo85A5X', 6), ('SCEa5CIg', 6)]
                # print("persons_top5:", persons_top5)

                # 每个正例采样5个负例
                # neg_author_list = random.sample(persons, 5)
                neg_author_list = [ pid[0] for pid in persons_top5]
                
                
                '''
                neg_author_list = random.sample(persons, neg_sample_num)

                for i in neg_author_list:
                    neg_ins.add((paper_id, paper_org, i))
                if len(neg_author_list) != neg_sample_num:
                    print("exception :", paper_id, paper_author_name, persons, neg_author_list)
            # if pos_num > 1:
            #     print("正例个数:", pos_num)
            # 获取同名的所有作者(除了本身)作为负例的candidate

            # if len(neg_author_list) < 5 :
            #     print("负例不足5个：",paper_id,now_author_name)

            # print("正负例个数:",paper_id,len(pos_ins),len(neg_ins))

            self.train_instances.append((pos_ins, neg_ins))

        print("训练样本长度：", len(self.train_instances))  # 500 10000
        # print("train_instances:",self.train_instances)
        print("saving train_instances ....")
        pickle.dump(self.train_instances, open(fout, "wb"))


    def refactor_train_instances(self,fn_train_instances,fn_new_instances='../data/instance/train_instance_new.pkl'):

        new_train_instances = []
        pre_train_instances = pickle.load(open(fn_train_instances,"rb"))
        for instance in pre_train_instances:

            new_pos_ins_set = set()
            new_neg_ins_set = set()


            pos_ins_set , neg_ins_set = instance
            # print("pos_ins_set:",pos_ins_set)
            # print("neg_ins_set:",neg_ins_set)

            for pos_ins in pos_ins_set :
                paper_id, paper_org, paper_author_id = pos_ins

                author_name = list(self.author_info[paper_author_id].keys())[0]
                if author_name in self.wrong_author_name_list:
                    continue

                new_paper_org = self.get_match_org0(paper_id, paper_author_id,paper_pub=self.paper_pub_train)
                new_pos_ins =  (paper_id, new_paper_org, paper_author_id)
                new_pos_ins_set.add( new_pos_ins )

            for neg_ins in neg_ins_set :
                paper_id, paper_org, paper_author_id = neg_ins
                author_name = list(self.author_info[paper_author_id].keys())[0]
                if author_name in self.wrong_author_name_list:
                    continue




                new_paper_org = self.get_match_org0(paper_id, paper_author_id,paper_pub=self.paper_pub_train)
                new_neg_ins =  (paper_id, new_paper_org, paper_author_id)
                new_neg_ins_set.add( new_neg_ins )

            if len(new_pos_ins_set) > 0:

                new_train_instances.append((new_pos_ins_set, new_neg_ins_set))

        pickle.dump(new_train_instances, open(fn_new_instances,"wb"))
        self.train_instances = new_train_instances

    def get_paper_coauthors(self, paper_pub, paperId, author_name):  # 获取当前论文除去当前作者以外的其他作者
        # print("in get_coauthors ......")
        # paper_pub = self.paper_pub_train
        paper_pub_cna = self.paper_pub_cna
        paper_coauthors = []

        author_list = []  # 来自于paper_pub
        paper_authors = []
        if paperId in paper_pub:
            paper_authors = paper_pub[paperId]['authors']
        else:
            paper_authors = paper_pub_cna[paperId]['authors']
        paper_authors_len = len(paper_authors)
        # 只取前50个author以保证效率
        paper_authors = random.sample(paper_authors, min(50, paper_authors_len))

        for author in paper_authors:
            clean_author = aligh_author_name(author['name'])
            if (clean_author != None and clean_author != ""):
                author_list.append(clean_author)  # 有同名的作者写同一篇论文的情况 没有处理

        if (len(author_list) > 0):
            # 获取paper中main author_name所对应的位置
            author_index = self.get_author_index(author_name, author_list)  # (['de feng zhang', 'zhong hong liang'], 2)
            # print("delete_main_name(author_list, paper_name):",delete_main_name(author_list, paper_name))
            # 获取除了main author_name外的coauthor
            # print("author_index:",author_index)
            for index in range(len(author_list)):
                if (index == author_index):
                    continue
                else:
                    paper_coauthors.append(author_list[index])

        # print("paper_coauthors:",paper_coauthors)
        return paper_coauthors

    def new_intersect(self, kws1, kws2):  # 关键词按空格分开的部分有重叠也算交集
        cnt = 0
        tot = 0
        for kw1 in kws1:
            words = set(kw1.split())  # [measurement precision]
            # fla
            for kw2 in kws2:
                words2 = set(kw2.split())  # [bearing-only measurement]
                if len(words & words2) > 0:
                    cnt += 1
                    # continue
                tot += 1
        if tot > 0:
            return cnt / float(tot)
        else:
            return -1




    def get_org_set(self,org):

        split_char = ",.&@()"
        for c in split_char:
            org = org.replace(c,' ')

        return set(org.split())

    def get_most_similar_year(self, paper_year, year_list): #有多个年份最相近未处理
        res_year = -1
        now_abs = 10000
        for year in year_list:
            if year != None and abs(year-paper_year) < now_abs:
                now_abs = abs(year-paper_year)
                res_year = year

        return res_year


    def get_org_simlarity(self,ins_affiliation,paper_year, authorId, mode):
        # 1.年份最相近 机构名特征
        #author_info_year.setdefault(authorId, {}).setdefault(author_name, {}).setdefault(tgt_year,set()).add(tgt_org)


        feature_list = []

        if paper_year == None:
            return [-1]

        #'Wtx5Pe3T': {'d_zhang': {2010: {'school mechanical engineering'}, 2011: {'school mechanical engineering'}, 2013: {'school mechanical engineering tianjin university'}}}


        org_info = self.author_info_year[authorId][list(self.author_info_year[authorId].keys())[0]] if authorId in self.author_info_year else None
        if org_info == None:
            return [-1]
        year_list = list(org_info.keys())
        # print("year_list:",year_list)


        most_sim_year = self.get_most_similar_year(paper_year,year_list)
        # print('most_sim_year:', most_sim_year)
        org_list = org_info[most_sim_year]
     # print("ins_affiliation:",ins_affiliation)

        levenshtein_dis = list()
        ratio = list()
        jaro_distance = list()
        jaccard_distance = list()

        if ins_affiliation != "" :  # 空字符串无意义 不比较

            for org in org_list:
                org = org.lower()
                levenshtein_dis.append(Levenshtein.jaro(org, ins_affiliation))
                seq = difflib.SequenceMatcher(lambda x: x in " \t \n", org, ins_affiliation)
                ratio.append(seq.ratio())  # 度量两个序列的相似程度 [0,1] 越大越相似

                jaro_distance.append(distance.get_jaro_distance(org, ins_affiliation, winkler=True, scaling=0.1))

                now_jac = self.jaccard(self.get_org_set(org), self.get_org_set(ins_affiliation))
                jaccard_distance.append(now_jac)
                # if now_jac == -1:
                #     print("jaccard error!:",org, ins_affiliation)
                #     print("instance:",instance)
                #     print("org_list:",org_list)

        if ins_affiliation != "" and len(levenshtein_dis) > 0 and len(ratio) > 0:

            max_leven = max(levenshtein_dis)
            max_ratio = max(ratio)
            max_jaro = max(jaro_distance)
            max_jaccard = max(jaccard_distance)

            # fout.write("1:%d " %(min_leven))
            if len(
                    levenshtein_dis) > 1 and mode == 'Train' and max_leven == 1.0 and max_ratio == 1.0 and max_jaro == 1.0:  # 负样本为1？
                sec_max_leven = sorted(levenshtein_dis, reverse=True)[1]
                sec_max_ratio = sorted(ratio, reverse=True)[1]
                sec_max_jaro = sorted(jaro_distance, reverse=True)[1]
                sec_max_jaccard = sorted(jaccard_distance, reverse=True)[1]

                # feature_list.extend([max_leven, max_ratio])

                feature_list.extend([sec_max_jaccard])  # sec_max_leven, sec_max_ratio,
            else:  # len = 1
                feature_list.extend([max_jaccard])  #max_leven, max_ratio,

        else:
            feature_list.extend([-1] )

        return feature_list
    def get_org_simlarity_by_year(self,ins_affiliation,paper_year, authorId, mode):
        # 1.年份最相近 机构名特征 -> 5年内
        #author_info_year.setdefault(authorId, {}).setdefault(author_name, {}).setdefault(tgt_year,set()).add(tgt_org)


        feature_list = []
        near_year_list = []

        org_info = self.author_info_year[authorId][list(self.author_info_year[authorId].keys())[0]] if authorId in self.author_info_year else None
        if org_info == None:
            return [-1] * 3

        if paper_year == None:
            pass
        else:
            #'Wtx5Pe3T': {'d_zhang': {2010: {'school mechanical engineering'}, 2011: {'school mechanical engineering'}, 2013: {'school mechanical engineering tianjin university'}}}
            year_list = list(org_info.keys())
            # print("year_list:",year_list)
            near_year_list = [ y for y in year_list if (y != None and abs(y-paper_year) <= 5)]
            # print("near_year_list:",near_year_list)

        if len(near_year_list) == 0:  # 没有相近的year 则比较全部年份
            near_year_list = list(org_info.keys())

        org_list = set()
        for year in near_year_list:
            org_list.update(org_info[year])

        levenshtein_dis = list()
        ratio = list()
        jaro_distance = list()
        jaccard_distance = list()

        if ins_affiliation != "" :  # 空字符串无意义 不比较

            for org in org_list:
                org = org.lower()
                levenshtein_dis.append(Levenshtein.jaro(org, ins_affiliation))
                seq = difflib.SequenceMatcher(lambda x: x in " \t \n", org, ins_affiliation)
                ratio.append(seq.ratio())  # 度量两个序列的相似程度 [0,1] 越大越相似

                jaro_distance.append(distance.get_jaro_distance(org, ins_affiliation, winkler=True, scaling=0.1))

                now_jac = self.jaccard(self.get_org_set(org), self.get_org_set(ins_affiliation))
                jaccard_distance.append(now_jac)
                # if now_jac == -1:
                #     print("jaccard error!:",org, ins_affiliation)
                #     print("instance:",instance)
                #     print("org_list:",org_list)

            # print("leven:",levenshtein_dis)
            # print("ratio:",ratio)

        if ins_affiliation != "" and len(levenshtein_dis) > 0 and len(ratio) > 0:

            max_leven = max(levenshtein_dis)
            max_ratio = max(ratio)
            max_jaro = max(jaro_distance)
            max_jaccard = max(jaccard_distance)

            # fout.write("1:%d " %(min_leven))
            if len(
                    levenshtein_dis) > 1 and mode == 'Train' and max_leven == 1.0 and max_ratio == 1.0 and max_jaro == 1.0:  # 负样本为1？
                sec_max_leven = sorted(levenshtein_dis, reverse=True)[1]
                sec_max_ratio = sorted(ratio, reverse=True)[1]
                sec_max_jaro = sorted(jaro_distance, reverse=True)[1]
                sec_max_jaccard = sorted(jaccard_distance, reverse=True)[1]

                # feature_list.extend([max_leven, max_ratio])

                feature_list.extend([sec_max_leven, sec_max_ratio,sec_max_jaccard])  # sec_max_leven, sec_max_ratio,
            else:  # len = 1
                feature_list.extend([max_leven, max_ratio,max_jaccard])  #max_leven, max_ratio,

        else:
            feature_list.extend([-1] * 3 )

        return feature_list

    def gen_triplet_title_feature(self, original_emb ,save_path='model/tm.feature'):
        triplet_model = TripletModel()
        triplet_model.load_state_dict(torch.load(os.path.join('model', 'tripletmodel.title.checkpoint')))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        triplet_model = triplet_model.to(device)

        original_emb0 = np.stack([pair[0].tolist() for pair in original_emb]) #(n,300)
        original_emb1 = np.stack([pair[1].tolist() for pair in original_emb])
        original_emb0 = np.expand_dims(original_emb0, axis=1)#(n,1，300）
        original_emb1 = np.expand_dims(original_emb1, axis=1)

        emb0 = torch.from_numpy(original_emb0).to(device).to(torch.float)
        emb1 = torch.from_numpy(original_emb1).to(device).to(torch.float)

        # emb0 = "0.27204 -0.06203 -0.1884 0.023225 -0.018158 0.0067192 -0.13877 0.17708 0.17709 2.5882 -0.35179 -0.17312 0.43285 -0.10708 0.15006 -0.19982 -0.19093 1.1871 -0.16207 -0.23538 0.003664 -0.19156 -0.085662 0.039199 -0.066449 -0.04209 -0.19122 0.011679 -0.37138 0.21886 0.0011423 0.4319 -0.14205 0.38059 0.30654 0.020167 -0.18316 -0.0065186 -0.0080549 -0.12063 0.027507 0.29839 -0.22896 -0.22882 0.14671 -0.076301 -0.1268 -0.0066651 -0.052795 0.14258 0.1561 0.05551 -0.16149 0.09629 -0.076533 -0.049971 -0.010195 -0.047641 -0.16679 -0.2394 0.0050141 -0.049175 0.013338 0.41923 -0.10104 0.015111 -0.077706 -0.13471 0.119 0.10802 0.21061 -0.051904 0.18527 0.17856 0.041293 -0.014385 -0.082567 -0.035483 -0.076173 -0.045367 0.089281 0.33672 -0.22099 -0.0067275 0.23983 -0.23147 -0.88592 0.091297 -0.012123 0.013233 -0.25799 -0.02972 0.016754 0.01369 0.32377 0.039546 0.042114 -0.088243 0.30318 0.087747 0.16346 -0.40485 -0.043845 -0.040697 0.20936 -0.77795 0.2997 0.2334 0.14891 -0.39037 -0.053086 0.062922 0.065663 -0.13906 0.094193 0.10344 -0.2797 0.28905 -0.32161 0.020687 0.063254 -0.23257 -0.4352 -0.017049 -0.32744 -0.047064 -0.075149 -0.18788 -0.015017 0.029342 -0.3527 -0.044278 -0.13507 -0.11644 -0.1043 0.1392 0.0039199 0.37603 0.067217 -0.37992 -1.1241 -0.057357 -0.16826 0.03941 0.2604 -0.023866 0.17963 0.13553 0.2139 0.052633 -0.25033 -0.11307 0.22234 0.066597 -0.11161 0.062438 -0.27972 0.19878 -0.36262 -1.0006e-05 -0.17262 0.29166 -0.15723 0.054295 0.06101 -0.39165 0.2766 0.057816 0.39709 0.025229 0.24672 -0.08905 0.15683 -0.2096 -0.22196 0.052394 -0.01136 0.050417 -0.14023 -0.042825 -0.031931 -0.21336 -0.20402 -0.23272 0.07449 0.088202 -0.11063 -0.33526 -0.014028 -0.29429 -0.086911 -0.1321 -0.43616 0.20513 0.0079362 0.48505 0.064237 0.14261 -0.43711 0.12783 -0.13111 0.24673 -0.27496 0.15896 0.43314 0.090286 0.24662 0.066463 -0.20099 0.1101 0.03644 0.17359 -0.15689 -0.086328 -0.17316 0.36975 -0.40317 -0.064814 -0.034166 -0.013773 0.062854 -0.17183 -0.12366 -0.034663 -0.22793 -0.23172 0.239 0.27473 0.15332 0.10661 -0.060982 -0.024805 -0.13478 0.17932 -0.37374 -0.02893 -0.11142 -0.08389 -0.055932 0.068039 -0.10783 0.1465 0.094617 -0.084554 0.067429 -0.3291 0.034082 -0.16747 -0.25997 -0.22917 0.020159 -0.02758 0.16136 -0.18538 0.037665 0.57603 0.20684 0.27941 0.16477 -0.018769 0.12062 0.069648 0.059022 -0.23154 0.24095 -0.3471 0.04854 -0.056502 0.41566 -0.43194 0.4823 -0.051759 -0.27285 -0.25893 0.16555 -0.1831 -0.06734 0.42457 0.010346 0.14237 0.25939 0.17123 -0.13821 -0.066846 0.015981 -0.30193 0.043579 -0.043102 0.35025 -0.19681 -0.4281 0.16899 0.22511 -0.28557 -0.1028 -0.018168 0.11407 0.13015 -0.18317 0.1323".split()



        triplet_model.eval()
        with torch.no_grad():
            print(emb0.size())  # torch.Size([1, 1, 300]) torch.Size([3, 1, 300])
            emb0 = triplet_model.get_emb(emb0)
            print(emb0.size())  # torch.Size([32]) torch.Size([3, 32])
            emb1 = triplet_model.get_emb(emb1)

        emb_distance = torch.sqrt(torch.sum(torch.pow((emb0 - emb1), 2), dim=1))
        emb_distance = emb_distance.cpu().numpy()
        df = pd.DataFrame(data=emb_distance)
        df.to_pickle(save_path)



    # paper_pub交给上层处理，train ->train_pub.json ,  test->whole_author_profile_pub.json
    def process_feature(self, paper_pub, author_json, instance, mode="Train"):  # author_json要根据train test mode来确定
        ## pos_ins:(paper_id, paper_author_id) ('opbhvKbU', 'LEjdcebe') #author_info:{author_Id:} paper_coauthors:[当前作者的合作者 name_list]
        ##author_json:train_author.json

        # global author_info  # author_info.setdefault(id,{}).setdefault(name,[]).extend(org_list)
        author_info = self.author_info

        paper_pub_cna = self.paper_pub_cna

        paperId = instance[0]
        ins_affiliation = instance[1].lower()
        ins_affiliation = char_filter(ins_affiliation) #将特殊字符都替换为空格 并转小写

        authorId = instance[2]
        author_name = list(author_info[authorId].keys())[0]

        feature_list = []
        # author_name = paper2aid2name[paperId][0][0]  # paper2aid2name[paper_id] =[ (author_name, person)]  paper2aid2name.setdefault(paper_id,[]).append ((author_name, author_id))

        # 年份最相近的机构相似度

        '''
        paper_year = None
        if paperId in self.paper_year:
            paper_year = self.paper_year[paperId]

        org_year_feature = self.get_org_simlarity(ins_affiliation, paper_year, authorId, mode)#def get_org_simlarity(self,ins_affiliation,paper_year, authorId, mode):
        # print('org_year_feature:',org_year_feature)

        feature_list.extend(org_year_feature)
        '''
        # new 1 -> 年份相近机构名特征
        # paper_year = None
        # if paperId in self.paper_year:
        #     paper_year = self.paper_year[paperId]
        #
        # org_year_feature = self.get_org_simlarity_by_year(ins_affiliation, paper_year, authorId,
        #                                                   mode)
        #
        # feature_list.extend(org_year_feature) #3



        # 1.机构名特征
        org_list = author_info[authorId][list(author_info[authorId].keys())[0]]
        author_name = list(author_info[authorId].keys())[0]
        levenshtein_dis = list()
        ratio = list()
        jaro_distance = list()
        jaccard_distance = list()

        if ins_affiliation != "": #空字符串无意义 不比较

            for org in org_list:
                org = org.lower()
                levenshtein_dis.append(Levenshtein.jaro(org, ins_affiliation))
                seq = difflib.SequenceMatcher(lambda x: x in " \t \n", org, ins_affiliation)
                ratio.append(seq.ratio())  # 度量两个序列的相似程度 [0,1] 越大越相似

                jaro_distance.append( distance.get_jaro_distance(org, ins_affiliation, winkler=True, scaling=0.1) )

                now_jac = self.jaccard(self.get_org_set(org), self.get_org_set(ins_affiliation))
                jaccard_distance.append(now_jac)
                # if now_jac == -1:
                #     print("jaccard error!:",org, ins_affiliation)
                #     print("instance:",instance)
                #     print("org_list:",org_list)

        if ins_affiliation != "" and len(levenshtein_dis) > 0 and len(ratio) > 0:

            max_leven = max(levenshtein_dis)
            max_ratio = max(ratio)
            max_jaro = max( jaro_distance)
            max_jaccard = max(jaccard_distance)

            # fout.write("1:%d " %(min_leven))
            if len(levenshtein_dis) > 1 and mode == 'Train' and max_leven == 1.0 and max_ratio ==1.0 and max_jaro == 1.0: #负样本为1？
                sec_max_leven = sorted(levenshtein_dis,reverse=True)[1]
                sec_max_ratio = sorted(ratio,reverse=True)[1]
                sec_max_jaro = sorted(jaro_distance,reverse=True)[1]
                sec_max_jaccard = sorted(jaccard_distance,reverse=True)[1]

            # feature_list.extend([max_leven, max_ratio])

                feature_list.extend([sec_max_leven, sec_max_ratio, sec_max_jaccard]) #
            else : #len = 1
                feature_list.extend([max_leven, max_ratio, max_jaccard]) #

        else:
            feature_list.extend([-1] * 3)

        #后面提取合作者机构特征需要
        paper_coauthors = self.get_paper_coauthors(paper_pub, paperId,
                                                   author_name)



        # 2. 合作者特征
        # 从作者的论文列表中把该篇论文去掉，防止训练出现bias
        doc_list = []  # 当前作者 除去当前论文的 所有论文
        paper_coauthors = self.get_paper_coauthors(paper_pub, paperId,
                                                   author_name)  ##根据论文元数据 获取当前论文除去当前作者以外的其他作者
        paper_coauthors_len = len( set(paper_coauthors) )

        if mode == "Train":
            """验证 author_name 是否正确"""
            # author_json = self.author_train
            if author_name not in author_json:
                print("error!姓名不匹配！：", author_name, authorId)

            for doc in author_json[author_name][authorId]:  # paper_name - author_name, author - authorId 获取当前作者的所有论文
                if (doc != paperId):
                    doc_list.append(doc)
            for doc in doc_list:
                if doc == paperId:
                    print("error!")
                    exit()
        elif mode == "Test":
            """验证 author_name 是否正确"""
            # author_json  = self.author_whole
            if authorId not in author_json:
                print("error!姓名不匹配！：", author_name, authorId)

            for doc in author_json[authorId]["papers"]:  # paper_name - author_name, author - authorId 获取当前作者的所有论文
                if (doc != paperId):
                    doc_list.append(doc)
            for doc in doc_list:
                if doc == paperId:
                    print("error!")
                    exit()
        # print("doc_list:",doc_list)

        # 保存当前作者的所有paper的coauthors以及各自出现的次数(作者所拥有论文的coauthors) === 当前作者的合作者
        candidate_authors_int = defaultdict(int)

        total_author_count = 0

        # print("instance:",instance)
        #通过author.json获取的 数值太稀疏！！！！！！！
        papers_with_coauthor = 0  # 当前论文的作者 与当前作者所有论文的作者 有交集的论文数目
        max_coauthor_num = 0
        doc_num_with_all_coauthor = 0
        doc_num_with_half_coauthor = 0
        # 无法根据author_name获取合作者所写的论文F
        for docId in doc_list:  ## 根据论文元数据 获取当前作者 除去当前论文的 所有论文id

            doc_dict = paper_pub[docId]  # train_pub.json
            author_list = []  # 当前作者的某篇论文的所有作者

            paper_authors = doc_dict['authors']
            paper_authors_len = len(paper_authors)
            """后面可尝试全部采样"""
            paper_authors = random.sample(paper_authors, min(50, paper_authors_len))

            for author in paper_authors:
                clean_author = aligh_author_name(author['name'])
                if (clean_author != None and clean_author != ""):
                    author_list.append(clean_author)

            # print("author_list:", author_list)#有同名作者写同一篇论文的如何处理？

            now_coauthor_len = len(set(author_list) & set(paper_coauthors))
            if  now_coauthor_len > 0:  # 排除掉当前作者 paper_coauthors就已经排除掉了
                papers_with_coauthor += 1
                max_coauthor_num = max(max_coauthor_num, now_coauthor_len)

            if now_coauthor_len == paper_coauthors_len:
                doc_num_with_all_coauthor += 1

            if now_coauthor_len >= paper_coauthors_len/2:
                doc_num_with_half_coauthor += 1


            if (len(author_list) > 0):
                # 获取paper中main author_name所对应的位置
                author_index = self.get_author_index(author_name,
                                                     author_list)  # paper_name:当前 paper的作者，author_name : 当前作者的name
                # print("get_author_index( author_name, author_list):", author_index)
                # 获取除了main author_name外的coauthor
                for index in range(len(author_list)):
                    if (index == author_index):
                        continue
                    else:
                        candidate_authors_int[author_list[index]] += 1
                        total_author_count += 1
        # print("paper_with_coauthors:",papers_with_coauthor)
        # author 的所有不同coauthor name
        author_keys = list(candidate_authors_int.keys())

        if ((len(author_keys) == 0) or (len(paper_coauthors) == 0)):
            feature_list.extend([0.] * 5)
        else:

            co_coauthors = set(paper_coauthors) & set(author_keys)  # paper_coauthors:当前论文其他作者   ，author_keys：当前作者的合作者
            coauthor_len = len(co_coauthors)

            # co_coauthors_ratio_for_paper = round(coauthor_len / len(paper_coauthors), 6)
            # co_coauthors_ratio_for_author = round(coauthor_len / len(author_keys), 6)

            co_coauthors_ratio_for_paper = coauthor_len / len(paper_coauthors)
            co_coauthors_ratio_for_author = coauthor_len / len(author_keys)

            coauthor_count = 0
            for coauthor_name in co_coauthors:
                coauthor_count += candidate_authors_int[coauthor_name]

            # co_coauthors_ratio_for_author_count = round(coauthor_count / total_author_count, 6)
            co_coauthors_ratio_for_author_count = coauthor_count / total_author_count

            # 计算了5维paper与author所有的paper的coauthor相关的特征：
            #    1. 不重复的coauthor个数
            #    2. 不重复的coauthor个数 / paper的所有coauthor的个数
            #    3. 不重复的coauthor个数 / author的所有paper不重复coauthor的个数
            #    4. coauthor个数（含重复）
            #    5. coauthor个数（含重复）/ author的所有paper的coauthor的个数（含重复）
            #    6.papers_with_coauthor / len(doc_list) (F -0.9)
            #   7.doc_num_with_all_coauthor/float(len(doc_list)) -0.004
            #   8. doc_num_with_half_coauthor/float(len(doc_list))
            #   6+8合成 -0.002
            feature_list.extend(
                [co_coauthors_ratio_for_paper, co_coauthors_ratio_for_author,
                 co_coauthors_ratio_for_author_count,max_coauthor_num/float(paper_coauthors_len),
                 papers_with_coauthor / len(doc_list)
                ])  #有交集的作者数


        #有交集的论文数:
        # 当前作者其余paper(paperId_list)  的作者(name_list) 与 当前paper的所有作者(name_list)有交集 的paper数量  （有共同合作者的paper）/ 当前作者的所有论文数量
        '''
       #         print(feature_list)
       # 3.合作者论文特征 -0.007 当前论文其他作者的论文 与 当前作者的论文 的jaccard
       #   跑验证集时paperId 很可能不在 self.paper_authors 中

       # overlap = list()
       # intersects = list()
       # if paperId in self.paper_authors:
       #     for coauthor in self.paper_authors[paperId]:
       #         if coauthor == authorId:
       #             continue
       #         # print('self.author_papers[authorId]:',self.author_papers[authorId])
       #         # print('self.author_papers[coauthor]:',self.author_papers[coauthor])
       #         union = len(self.author_papers[authorId] | self.author_papers[coauthor])
       #         intersect = len(self.author_papers[authorId] & self.author_papers[coauthor])
       #         intersects.append(intersect)
       #         overlap.append(float(intersect) / float(union))
       #     if len(overlap) > 0:
       #         feature_list.append((max(overlap)))
       #         feature_list.append((max(intersects)))
       #         # print('overlap,intersects:',overlap,intersects)
       #     else :
       #         feature_list.extend([0.] * 2)
       # else:
       #     feature_list.extend([0.] * 2)


       '''

        '''
        # 4.时间跨度特征(-0.009 -> +0.002)
        # paper_year = self.paper
        # print("instance:",instance)
        # farestpaperyear = -1
        # print('self.paper_year[paperId]:', self.paper_year[paperId])
        if (paperId  in self.paper_year) and self.paper_year[paperId] != ""  and self.paper_year[paperId] != 0 and self.paper_year[paperId] != 210 and authorId in self.author_year_avg:
            yrscore = abs(self.paper_year[paperId] - self.author_year_avg[authorId])


            # for otherpaper in doc_list:
            #     # print(otherpaper)
            #     if otherpaper == paperId or (otherpaper not in self.paper_year) or self.paper_year[otherpaper] == ""  or self.paper_year[otherpaper] == 0:
            #         continue
            #     if otherpaper in self.paper_year:
            #         # print('self.paper_year[otherpaper]:',self.paper_year[otherpaper])
            #         if abs(self.paper_year[paperId] - self.paper_year[otherpaper]) > farestpaperyear:
            #             farestpaperyear = abs(self.paper_year[paperId] - self.paper_year[otherpaper])
            # yrscore = self.year_range_score[farestpaperyear] if farestpaperyear in self.year_range_score else self.year_range_score["unknown"]
            feature_list.append(float(yrscore)/10 )
        else:

            feature_list.append(-1)
        '''





        # 5.当前作者其他论文的标题 与 当前论文标题的jaro_distance levenshtein_distance

        levenshtein_dis = list()
        ratio = list()
        title_jaccard = list()

        paper_title = self.paper_title[paperId] if paperId in self.paper_title else ""
        # print("instance:",instance)
        # print("paper_title:",paper_title)
        # print("".join(paper_title.split()))
        for docId in doc_list:
            if docId in self.paper_title: # 加载数据时 已经排除了空字符串
                # print("self.paper_title[docId]:",self.paper_title[docId])

                # print("".join(self.paper_title[docId].split()))
                # print("leven:", Levenshtein.jaro(paper_title, self.paper_title[docId]))

                levenshtein_dis.append(
                    Levenshtein.jaro(paper_title, self.paper_title[docId]) )  # 值越高越相似
                ratio.append(difflib.SequenceMatcher(lambda x: x in " \t \n", paper_title,
                                                     self.paper_title[docId]).ratio())

                now_jac = self.jaccard(self.get_org_set(paper_title), self.get_org_set(self.paper_title[docId]))
                title_jaccard.append(now_jac)

        if paper_title != "" and len(levenshtein_dis) > 0 and len(ratio) > 0:

            max_leven = max(levenshtein_dis)
            max_ratio = max(ratio)
            max_jaccard = max(title_jaccard)
            # fout.write("1:%d " %(min_leven))
            feature_list.extend([max_leven, max_ratio, max_jaccard])
        else:
            feature_list.extend([-1] * 3)
            



        # 6.keywords信息
        # print("instance:",instance)
        intersects = []
        # print("instance:",instance)

        if (paperId in self.paper_keywords) and self.paper_keywords[paperId] != None and len(
                self.paper_keywords[paperId]) != 0:
            for docId in doc_list:
                # if ind >1 :
                #     break
                # print("docId:",docId)
                if docId in self.paper_keywords:

                    intersect = len( set(self.paper_keywords[paperId] ) & set(self.paper_keywords[docId]))
                    union = len(set(self.paper_keywords[paperId] ) | set(self.paper_keywords[docId] ) )

                    # print("self.paper_keywords[paperId] :",self.paper_keywords[paperId] )
                    # print("self.paper_keywords[docId]:",self.paper_keywords[docId])
                    # print(intersect,union,float(intersect) / float(union))

                    if union > 0:
                        # intersects.append(self.new_intersect(self.paper_keywords[paperId], self.paper_keywords[docId]))
                        intersects.append(float(intersect) / float(union))
        if len(intersects) > 0:
            feature_list.append(max(intersects))
        else:
            feature_list.append(-1)



        # 7.abstract
        # print("instance:",instance)
        abstract_jaccard = list()
        # levenshtein_dis_abstract = list()
        if (paperId in self.paper_abstract) and self.paper_abstract[paperId] != None and len(self.paper_abstract[paperId]) != 0:
            paper_abstract = self.paper_abstract[paperId]
            for docId in doc_list:
                # print("docId:",docId)
                if docId in self.paper_abstract:

                    now_jac = self.jaccard(self.get_org_set(paper_abstract), self.get_org_set(self.paper_abstract[docId]))
                    abstract_jaccard.append(now_jac)
                    # print(self.paper_abstract[docId])
                    # print(paper_abstract)
                    # print(Levenshtein.jaro(self.paper_abstract[docId],paper_abstract))

                    # levenshtein_dis_abstract.append(Levenshtein.jaro(self.paper_abstract[docId],paper_abstract))

        if len(abstract_jaccard) != 0:
            feature_list.append( max( abstract_jaccard ) )
        else:
            feature_list.append(-1)
            

        # 8.venue 测试集几乎全为-1 太稀疏 换成相似度
        # if paperId in self.paper_venue and authorId in self.author_venue:
        #     now_paper_venue = self.paper_venue[paperId]
        #     now_author_venue = self.author_venue[authorId]
        #     if now_paper_venue in now_author_venue:
        #         feature_list.append(now_author_venue[now_paper_venue])
        #     else:
        #         feature_list.append(-1)
        # else:
        #     feature_list.append(-1)
        '''
        levenshtein_dis_venue = list()
        ratio_venue = list()

        if paperId in self.paper_venue and authorId in self.author_venue:
            now_paper_venue = self.paper_venue[paperId]
            now_author_venue = list(self.author_venue[authorId].keys())
            for v in now_author_venue:
                levenshtein_dis_venue.append(Levenshtein.jaro(now_paper_venue,v))
                ratio_venue.append(difflib.SequenceMatcher(lambda x: x in " \t \n",now_paper_venue,v).ratio())

        if len(levenshtein_dis_venue) > 0 and len(ratio_venue) > 0:
            feature_list.extend( [max(levenshtein_dis_venue),max(ratio_venue)] )
        else:
            feature_list.extend([-1]*2)
        '''



        # 10.当前paper其余作者机构 与当前作者机构的相似度信息
        # print("instance:", instance )
        now_paper_pub = paper_pub
        if mode == "Test":
            now_paper_pub = self.paper_pub_cna
        other_authors = now_paper_pub[paperId]["authors"]
        levenshtein_dis_org = list()
        ratio_org = list()
        org_jaccard_distance = list()



        # print("instance：",instance)
        org_list = self.author_info[authorId][list(author_info[authorId].keys())[0]] #当前作者机构列表
        author_name = list(author_info[authorId].keys())[0]

        coauthors_with_same_org = 0
        other_author_tot = 0

        # print("paper_coauthors:",paper_coauthors)
        for oauthor in other_authors:
            other_org = char_filter( oauthor["org"] ) if "org" in oauthor else ""
            name = oauthor["name"] if "name" in oauthor else ""

            clean_name = aligh_author_name(name)
            if clean_name not in paper_coauthors: # # 把当前作者去掉避免出现偏差
                continue

            # if aligh_author_name(name) or get_converse_name( name )== author_name:  # 把当前作者去掉避免出现偏差
            #     continue
            # print("other:", oauthor)
            similar_flag = False
            other_author_tot += 1



            if other_org != "" :
                # other_org = other_org.lower()
                # other_org = char_filter(other_org)
                # if other_org != ins_affiliation : #去除和目标作者相同的机构名

                    for org in org_list:

                        org = org.lower()
                        # print(other_org,org,Levenshtein.jaro(org, ins_affiliation),seq.ratio())

                        levenshtein_dis_org.append(Levenshtein.jaro(org, other_org))
                        seq = difflib.SequenceMatcher(lambda x: x in " \t \n", org, other_org)
                        ratio_org.append(seq.ratio())  # 度量两个序列的相似程度 [0,1] 越大越相似

                        now_jac = self.jaccard(self.get_org_set(org), self.get_org_set(other_org))
                        org_jaccard_distance.append(now_jac)
                        if now_jac == -1:
                            print("jaccard error!:", org, other_org)
                        elif  now_jac > 0.6:
                            similar_flag = True
                            # print("similar:")
                            # print('other_org:',other_org)
                            # print('org:',org)

                # elif other_org == ins_affiliation :
                #     for org in org_list:
                #
                #         org = org.lower()
                #         # print(other_org,org,Levenshtein.jaro(org, ins_affiliation),seq.ratio())
                #
                #
                #         now_jac = self.jaccard(self.get_org_set(org), self.get_org_set(other_org))
                #         # org_jaccard_distance.append(now_jac)
                #         if now_jac == -1:
                #             print("jaccard error!:", org, other_org)
                #         elif  now_jac > 0.6:
                #             similar_flag = True
                #             # print("similar:", other_org)



            if similar_flag:
                 coauthors_with_same_org += 1
                    # print(self.get_org_set(org),'\n', self.get_org_set(other_org))
                    # print(other_org, org, Levenshtein.jaro(org, ins_affiliation), seq.ratio(), self.jaccard(self.get_org_set(org), self.get_org_set(other_org)))
        # print("coauthors_with_same_org:",coauthors_with_same_org)
        if len(levenshtein_dis_org) > 0 and len(ratio_org) > 0:

            max_leven = max(levenshtein_dis_org)
            max_ratio = max(ratio_org)
            # fout.write("1:%d " %(min_leven))
            max_jaccard = max(org_jaccard_distance)
            feature_list.extend([max_leven, max_ratio, max_jaccard])
        else:
            feature_list.extend([-1] * 3)

        if other_author_tot > 0 :
            feature_list.append( coauthors_with_same_org/float(other_author_tot) )
        else:
            feature_list.append(-1)
            



        '''
        print("instance:", instance)
        # 11 当前论文摘要 与当前作者 所有论文摘要的最大tf-idf相似度
        max_sim = -1
        if paperId in self.paper_abstract:
            p_abs = self.paper_abstract[paperId]

            if authorId in self.author_sim_model:
                sim_model = self.author_sim_model[authorId]
                max_sim = sim_model.get_max_similarity(p_abs)

        feature_list.append(max_sim)
        
        '''

        '''
        # 18.作者拥有论文数
        author_paper_num = -1

        if authorId in self.author_papers:
            author_paper_num = len(self.author_papers[authorId]) / float(100)

        feature_list.append(author_paper_num)
        '''

        '''
        # 19.title abstract keywords -> all_keywords -0.02

        intersects_all_kws = []

        if (paperId in self.paper_all_keywords) and self.paper_all_keywords[paperId] != None and len(
                self.paper_all_keywords[paperId]) != 0:
            for docId in doc_list:
                # if ind >1 :
                #     break
                # print("docId:",docId)
                if docId in self.paper_all_keywords:

                    intersect = len(self.paper_all_keywords[paperId] & self.paper_all_keywords[docId])
                    union = len(self.paper_all_keywords[paperId] | self.paper_all_keywords[docId])

                    # print("self.paper_all_keywords[paperId] :",self.paper_all_keywords[paperId] )
                    # print("self.paper_all_keywords[docId]:",self.paper_all_keywords[docId])
                    # print(intersect,union,float(intersect) / float(union))

                    if union > 0:
                        # intersects.append(self.new_intersect(self.paper_keywords[paperId], self.paper_keywords[docId]))
                        intersects_all_kws.append(float(intersect) / float(union))
        if len(intersects_all_kws) > 0:
            feature_list.append(max(intersects_all_kws))
        else:
            feature_list.append(-1)

        '''

        #20 title_bert_precision
        paper_info = self.paper_pub_whole[paperId]
        title = ""

        if "title" in paper_info and (paper_info["title"] != None) and (len(paper_info["title"]) > 0):
            title = paper_info["title"]

        # print(title)

        paperIds = self.author_whole[authorId]["papers"]
        # print(paperIds)

        now_corpus = ""

        for pid in paperIds:
            if pid != paperId:
                now_paper = self.paper_pub_whole[pid]

                if "title" in now_paper and (now_paper["title"] != None) and (len(now_paper["title"]) > 0):
                    now_corpus += now_paper["title"]
        pred = 1
        if title != "" and now_corpus != "":
            row = (title, now_corpus)
            tokens = convert_data(row)
            pred = bert_model(tokens)

        feature_list.append(pred)


        return feature_list

    # 只有train的时候才调用此函数
    def gen_feature(self, fn_feature):  # author_info:{author_Id:}

        author_json = self.author_train
        paper_json = self.paper_pub_train

        pos_features = []
        neg_features = []
        # global paper2aid2name
        author_info = self.author_info
        train_instances = self.train_instances
        print('train_instances len :', len(train_instances))  # 500
        '''
        train_instances:
        [({('opbhvKbU', 'LEjdcebe')}, {('opbhvKbU', 'll03UsNU')}), ({('yXTlHZ1L', '78UYqrGc')}, {('yXTlHZ1L', '8ZF0wml5')}), ({('xiSPtlti', 'HnwkZtEg')}, {('xiSPtlti', 'ObKkeRL0')}), ({('Hx7nKu0p', 'Gjmb7WuI')}, {('Hx7nKu0p', '8vLn5Q7O')}), ({('mTNsNZir', 'LLSJqSR3')}, {('mTNsNZir', 'qiadKgDB')}), ({('dsunvB9U', '9IS4TkP6')}, {('dsunvB9U', 'A21hiK8v')}), ({('PIF20Mla', 'OpirwsAm')}, {('PIF20Mla', 'THoMiFfW')}), ({('z8wIrQ4s', 'flshajOb')}, {('z8wIrQ4s', 'sRGEGq69')}), ({('XWDvbCTB', 'f0XXmG2S')}, {('XWDvbCTB', 'S5pmuEOl')}), ({('HuafyGC0', 'Of2shY0g')}, {('HuafyGC0', 'fpZRIaVq')})]

        '''

        '''把每个作者的正负例取出来'''
        with open(fn_feature, "w", encoding="utf-8")as fout:
            for ins in tqdm( train_instances ):  # 获取合作者

                pos_set = ins[0]
                neg_set = ins[1]
                paper_id = list(pos_set)[0][0]
                # paper_name = paper2aid2name[paper_id][0]

                author_list = []
                # if len(pos_set) >1

                for pos_ins in pos_set:
                    # process_feature(author_info,author_json,paper_pub, instance, paper_coauthors):
                    ins = self.process_feature(author_json=author_json, paper_pub=paper_json,
                                               instance=pos_ins, mode="Train")
                    pos_features.append(ins
                                        )  # pos_ins:(paper_id, paper_author_id) ('opbhvKbU', 'LEjdcebe')
                    fout.write(" ".join(map(lambda x: str(x), ins)))
                    fout.write(" 1\n")

                for neg_ins in neg_set:
                    ins = self.process_feature(author_json=author_json, paper_pub=paper_json,
                                               instance=neg_ins, mode="Train")
                    neg_features.append(ins)
                    fout.write(" ".join(map(lambda x: str(x), ins)))
                    fout.write(" 0\n")
            # 构建svm正负例
            svm_train_ins = []
            for ins in pos_features:
                svm_train_ins.append((ins, 1))

            # print("pos example:", svm_train_ins[0])
            for ins in neg_features:
                svm_train_ins.append((ins, 0))

            # print("neg example:", svm_train_ins[-1])
            print("训练样本大小:", np.array(svm_train_ins).shape)  # (30377, 2)
            # print(svm_train_ins)

            '''
            # 获取paper的coauthors
            paper_coauthors = []

            paper_authors = pubs_dict[paper_id]['authors']
            paper_authors_len = len(paper_authors)
            # 只取前50个author以保证效率
            paper_authors = random.sample(paper_authors, min(50, paper_authors_len))

            for author in paper_authors:
                clean_author = clean_name(author['name'])
                if (clean_author != None):
                    author_list.append(clean_author)

            if (len(author_list) > 0):
                # 获取paper中main author_name所对应的位置
                _, author_index = delete_main_name(author_list, paper_name)

                # 获取除了main author_name外的coauthor
                for index in range(len(author_list)):
                    if (index == author_index):
                        continue
                    else:
                        paper_coauthors.append(author_list[index])

                for pos_ins in pos_set:
                    pos_features.append(process_feature(pos_ins, paper_coauthors))

                for neg_ins in neg_set:
                    neg_features.append(process_feature(neg_ins, paper_coauthors))
            '''

    '''
gen_test_instance('data/cna_valid_unass_competition.json',
                                                         'data/whole_author_profile_pub.json',
                                                         'data/cna_valid_pub.json', 'data/whole_author_profile.json',
                                                         "data/all_author_info.txt", 'data/feature/feature_predict_1.txt',
                                                         write=True)

    '''

    def match_exception_name(self,paperId,author_name):
        # 针对测试集的姓名对应表
        res = None

        if paperId == 'CXAJ11nV' and author_name == "hang":
            res = 'hang_li'
        elif paperId =="JRUqE63F" and author_name == "jun":
            res = 'jun_yang'
        elif paperId == "gBa5zgUl" and author_name == "bo":
            res = "bo_li"
        elif  paperId == "w8AUc8xI" and author_name == "jianping":
            res = "jianping_ding"
        elif paperId == "N31mMo3Z" and author_name == "fei":
            res = "fei_wei"
        elif paperId == "wRlPD2yM" and author_name == "lin": #
            res = "lin_he"
        elif paperId == "xP7KjsKl" and author_name == "jinghong": #xP7KjsKl jinghong
            res = "jinghong_li"
        elif paperId == "52mWy7mz" and author_name == "jianfang": #52mWy7mz jianfang
            res = "jianfang_wang"
        elif paperId == "scA3nGBA" and author_name == "hui": #scA3nGBA hui
            res = "hui_zhang"
        elif author_name == "osamu":
            res = "osamu_watanabe"


        return res


    def match_abbrev_name(self,author_name):
        name_list = self.whole_name_list

        res_name = None
        match_flag = False

        for name in name_list:  # Li Guo

            if is_same_abbrev_name(author_name, name):  # "wang",
                match_flag = True


                if res_name == None:
                        res_name = name
                elif res_name != name:  # 有多个简称匹配
                        print('abbrev match error!', author_name )
                        return None

                        # final_name = match_name_by_score(author_name, author_list)
                        # print("final_o:",final_o)

                        # if "org" in final_o and final_o["org"] != "":
                        #     match_org = final_o["org"]
                        #     # print("final_org:",match_org)
                        #     if final_o["name"] == 'Luo Jiayan' and author_name == 'jie_luo':
                        #         match_org = None




        return res_name


    def gen_test_instance(self, fn_author_predict, fout_feature_test,
                          write=True):
        # fn_paper =
        author_info_name_key = self.author_name_ids
        # fn_author_info)  # author_info.setdefault(id,{}).setdefault(name,[]).extend(org_list)
        author_info = self.author_info
        # 存储paper的所有candidate author id
        paper2candidates = defaultdict(list)
        # 存储对应的paper与candidate author的生成特征
        paper2features = defaultdict(list)
        none_match_list = []



        with open(fn_author_predict, "r") as author_list, open(fout_feature_test, "w", encoding="utf-8") as fout:

            author_list = json.load(author_list)
            print("unass_list_len:", len(author_list))
            papers = self.paper_pub_whole
            papers_cna = self.paper_pub_cna

            # print(author_list)
            author_json = self.author_whole

            whole_name_list = list( author_info_name_key.keys() )

            for string in tqdm( author_list ):
                paperId, auId = string.split("-")
                auId = int(auId)

                # author = -1
                # if paperId in papers:
                author = papers_cna[paperId]["authors"][auId]
                # print(author)
                author_name = author["name"]  # 根据name查候选
                author_org = author["org"] if "org" in author else ""
                # print("待匹配：",paperId, author_name, author_org)

                authors_cands = []

                if author_name[-1] == '.':
                    author_name = author_name[:-1]
                author_name = author_name.replace("Dr.","").strip()
                author_name = author_name.replace(',',"")

                if author_name.find(" ") != -1 or author_name.find(
                        "\u00a0") != -1:  # Lin-Qi Jianmin\u00a0Zhao == jianmin_zhao

                    if aligh_author_name(author_name) in author_info_name_key:
                        authors_cands.extend(author_info_name_key[aligh_author_name(author_name)])

                    # author_name = aligh_author_name(author_name)

                    if get_converse_name(author_name) in author_info_name_key:
                        # print(get_converse_name(author_name))#Yang Jie -> jie_yang
                        # print("error:",author_name,paperId)
                        authors_cands.extend(author_info_name_key[get_converse_name(author_name)])

                else:
                    # print("author_name without space:", paperId, author_name)
                    pass

                author_cands = list(set(authors_cands))

                if len(authors_cands) == 0:
                    rule_match_name = self.match_exception_name(paperId,author_name)
                    if rule_match_name != None:
                        authors_cands.extend(author_info_name_key[rule_match_name])
                    else:
                        # abbrev_match_name = self.match_abbrev_name(author_name)
                        # if abbrev_match_name != None:
                        #     print("match by abbrev:",author_name, abbrev_match_name)
                        #     authors_cands.extend(author_info_name_key[abbrev_match_name])
                        # else:

                        print("no match candidate paperId:", paperId, author_name)  # Jie Yang 0002无法匹配
                        none_match_list.append(string)

                for candId in authors_cands:
                    # process_feature_predict(author_info,author_json,paper_pub, paper_pub_cna, instance):
                    # process_feature(author_info=author_info, author_json=author_json, paper_pub= paper_json,paper_pub_cna=paper_pub_cna, instance=pos_ins)
                    # process_feature(self, paper_pub ,author_json, instance, mode="Train"):
                    fea = self.process_feature(author_json=author_json, paper_pub=papers,
                                               instance=(paperId, author_org, candId), mode="Test")
                    # fea = [0] * 17
                    # print(fea)
                    # print(list(map(lambda x: str(x), fea)))
                    paper2candidates.setdefault(paperId, []).append(candId)
                    paper2features.setdefault(paperId, []).append(fea)
                    if write:
                        fout.write(" ".join(list(map(lambda x: str(x), fea))))
                        fout.write("\n")
        # print(none_match_list)
        return paper2candidates, paper2features

def gen_predict_feature():
        print("==================================================")
        print("gen predicting feature ...")

        dc = DataCollection(
            fn_author_info="data/all_author_info.txt",
            fn_author_info_year="data/all_author_info_with_year.pickle",
            # fn_author_info_paperid="data/all_author_info_with_paperid.pickle",
            fn_train_pub='data/train_pub.json',
            fn_train_author="data/train_author.json",
            fn_whole_author='data/whole_author_profile.json',
            fn_whole_pub='data/whole_author_profile_pub.json',
            fn_valid_pub="data/cna_test_pub.json")  # data/cna_valid_pub.json



        paper2candidates, paper2features = dc.gen_test_instance('data/cna_valid_unass_competition.json',
                                                                'data/feature/feature_predict.txt',
                                                                write=True)

        pickle.dump(paper2candidates, open('model/paper2candidates.pkl',"wb"))
        pickle.dump(paper2features, open('model/paper2features.pkl',"wb"))


        assert len(paper2candidates) == len(paper2features)


if __name__ == "__main__":


    paper_pub = {}
    author_json = {}
    with open('../data/train_pub.json', "r") as papers, open("../data/train_author.json", "r") as authors:
        paper_pub = json.load(papers)
        author_json = json.load(authors)

    dc = DataCollection()
    dc.gen_train_instances(11000,2)

    # dc.load_train_instances()

    # dc.refactor_train_instances('../data/instance/train_instance.pkl') #重新匹配org

    dc.gen_feature("../data/feature/feature_list_train.txt")

    gen_predict_feature()
