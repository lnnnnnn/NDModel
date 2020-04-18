import json
import re
from pyjarowinkler import distance
import numpy as np
import nltk,pickle
from collections import defaultdict




def char_filter( str):
    #\ufffd\ufffd
    stoplist = nltk.corpus.stopwords.words('english')
    stoplist.extend(['email', 'e-mail', 'address', 'fax']) #'city'

    str = str.lower()
    str = str.replace('\\u202f', ' ')
    str = str.replace('\\u2212', '−')
    str = str.replace('\\u2013', '-')  # –
    str = str.replace('\\u2218', '.')  # ∘
    # str = str.replace('\u03bc', ' ')#μ $$$$ \u03b3

    str = str.replace('\\u2019', '’')
    str = str.replace('\\u00a0', ' ')
    str = str.replace('\\u00b0', ' ') #°
    str = str.replace('&#x', ' ')
    # str = str.replace(" and ", ' & ') # ' and '
    str = str.replace('&amp;', '&')
    str = str.replace('%%#', ' ')
    str = str.replace('~', ' ')
    str = str.replace('$$$$', ' ')
    str = str.replace('^', ' ')
    str = str.replace('#',' ')
    str = str.replace('&quot;', '"')
    str = str.replace('‡', ' ')

    str = str.replace('&', " and ")
    str = str.replace("*"," ")
    str = str.replace("/"," ")




    #########################
    str = str.replace("these authors contributed equally to this work","")
    # "Wei Chen 0001"
    str = " ".join([w for w in str.replace(",", ' ').replace(".", ' ').replace("|", ' ').replace(";", ' ').replace(":", ' ').replace("(", ' ').replace(")",' ').replace( "<SUP>", "").replace("</SUP>", "").split() if not w in stoplist]) #:.replace(":", ' ')

    str = str.strip()# "." -> 导致空 把两端空格去掉
    '''
    %%#%%#~*'''

    return str





def get_author_set(arr):
    arrj = list(map( lambda x : json.dumps(x), arr)) #把json转为string 去重
    arrj = set(arrj)
    arrj = list(map(lambda x: json.loads(x), arrj))
    return arrj
def is_same_abbrev_name(name1,name2):
    return get_abbrev_name(name1) == get_abbrev_name(name2)

def get_abbrev_name(name): #hong_li

    rare_name_dict = {
        "guochun_zhao": "guo_chun_zhao",
        "Guochun C. Zhao": "Guo_chun Zhao",
        "weidong_li": "wei_dong_li",
        "B.i.n. Zhang": 'bin_zhang',
        "jianguo_hou": "jian_guo_hou",
        "haiying_wang": "hai_ying_wang",
        "jianping_ding": "jian_ping_ding",
        "H ONG L I": " hong_li",
        "jianping_fan": "jian_ping_fan",
        "Jianping Jianping Fan": "jian_ping_fan",
        "Rakesh (Teddy) Kumar": "rakesh_kumar"

    }

    name = re.sub(r'000[0-9]', "", name)
    name = re.sub(u"[\u4e00-\u9fa5]", '', name)   #\u5468\u6d9b  "name": "Jun Chen \u9648\u4fca", re.sub(u"[\u4e00-\u9fa5]", '', "Jun Chen \u9648\u4fca")
    if name in rare_name_dict:
        name = rare_name_dict[name]

    name = name.replace("_"," ")
    cleaned_name = re.sub(r'\W', ' ', name).lower().split()

    abbrevname = " ".join( sorted([letter[0] for letter in cleaned_name]) )
    return abbrevname

def aligh_author_name(name):

    '''
LIU Chang-ming == changming_liu
    c_h_chen == C.-H. Chen
C.H. Chen ==(?) c_h_chen
A-Young Kim
SHEN Xiao-mei

Zhi-Gang Zhang  == zhi_gang_zhang ???
"Hong-Yan Wang"  == hong_yan_wang

"Wei Chen 0001"
    '''
    # 需要替换 unicode为空么？
    name = name.replace("\u00a0"," ") #  bo_shen Bo\u00a0Shen -> Bo Shen


    name = re.sub(r'000[0-9]', "", name)

    name = name.strip()
    if len(name.split(" ")) > 2: #Jie Yang 0002
        # print("space more than 2 :",name)
        last_name = name.split(" ")[-1]
        other_name = "".join(name.split(" ")[0:-1])
        name = " ".join((other_name,last_name))
        # print( " after concate: ",name )
    name = name.replace(", ", "_")  # "Jianping, Fan"

    if name.find(".") != -1:
        name = name.lower().replace(". ", "_").replace(".-","_").replace(".","_")
    elif name.find("-") != -1:
        name = name.lower().replace(" ", "_").replace("-","")
    else:
        name = name.lower().replace(" ", "_")



    return name



#Shen Bo == bo_shen
def get_converse_name(name):
    # print("converse name:",name)
    name = name.replace("\u00a0", " ")
    # align = aligh_author_name(name)
    le, ri = name.split(" ")[:2]#{'name': 'Jie Yang 0002', 'org': ''}
    conv_name = " ".join((ri,le))
    return aligh_author_name(conv_name)

'''
            {
                "name": "long"
            }, 缺org 不算none match
'''


def clean_name(name):
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


def match_org_by_score(author_name, author_list):
    score_list = []
    name = clean_name(author_name)
    # author_list_lower = []
    # for author in author_list:
    #     author_list_lower.append(author.lower())

    # author_list_clean = list(map(clean_name, author_list))
    # print("author_list_clen:",author_list_clean)
    name_split = name.split()
    for o in author_list:
        if "name" in o and o["name"] != "":

            author = clean_name( o["name"] )

            # lower_name = author.lower()
            score = distance.get_jaro_distance(name, author, winkler=True, scaling=0.1)
            author_split = author.split()
            inter = set(name_split) & set(author_split)
            alls = set(name_split) | set(author_split)
            score += round(len(inter) / len(alls), 6)
            score_list.append(score)

    rank = np.argsort(-np.array(score_list))
    return_list = [author_list[i] for i in rank]

    return return_list[0]

def get_substr_org(author_name, author_list): #di_wang

    match_org = None
    match_flag = False
    for o in author_list:  # Li Guo
        if "name" in o and  author_name.lower().find( o["name"].replace(" ","").lower()) != -1: #"wang",   " Tang",

            match_flag = True
            if "org" in o and o["org"] != "":
                org = o["org"]

                org = org.replace("\n", '\\n').replace("\t", " ")
                org = char_filter(org)
                # global author_info
                if match_org == None :
                    match_org = org
                elif match_org != org:
                    print('org error!',author_name, author_list)

    return match_org, match_flag

'''
def get_substr_org_year(author_name, author_list): #di_wang

    match_org = None
    match_year = None
    match_flag = False
    for o in author_list:  # Li Guo
        if "name" in o and  author_name.lower().find( o["name"].replace(" ","").lower()) != -1: #"wang",   " Tang",

            match_flag = True
            if "org" in o and o["org"] != "":
                org = o["org"]

                org = org.replace("\n", '\\n').replace("\t", " ")
                org = char_filter(org)
                # global author_info
                if match_org == None :
                    match_org = org
                    if "year" in o and o["year"] != "":
                        match_year = o["year"]
                elif match_org != org:
                    print('org error!',author_name, author_list)

    return match_org, match_year,  match_flag
'''

def get_abbrevname_match_org(author_name, author_list):
    match_org = None
    match_flag = False
    org_all_null = True
    for o in author_list:  # Li Guo
        if "org" in o and o["org"] != "":
            org_all_null = False

        if "name" in o and   is_same_abbrev_name( author_name,  o["name"]) :  # "wang",
            match_flag = True


            if "org" in o and o["org"] != "":

                org = o["org"]

                org = org.replace("\n", '\\n').replace("\t", " ")
                org = char_filter(org)
                # global author_info
                if match_org == None:
                    match_org = org
                elif match_org != org: #有多个简称匹配
                    print('abbrev org error!', author_name, author_list)

                    final_o = match_org_by_score(author_name,author_list)
                    # print("final_o:",final_o)

                    if "org" in final_o and final_o["org"] != "":
                        match_org = final_o["org"]
                        # print("final_org:",match_org)
                        if final_o["name"] == 'Luo Jiayan' and author_name == 'jie_luo':
                            match_org = None

            # if o["name"] =="Jun Chen \u9648\u4fca":
            #     print("Jun Chen \u9648\u4fca", match_org)


    if match_flag == False and org_all_null:
        match_flag = True

    return match_org, match_flag
'''
def get_abbrevname_match_org_year(author_name, author_list): #函数作废
    match_org = None
    match_year = None
    match_flag = False
    org_all_null = True
    for o in author_list:  # Li Guo
        if "org" in o and o["org"] != "":
            org_all_null = False

        if "name" in o and   is_same_abbrev_name( author_name,  o["name"]) :  # "wang",
            match_flag = True


            if "org" in o and o["org"] != "":

                org = o["org"]

                org = org.replace("\n", '\\n').replace("\t", " ")
                org = char_filter(org)
                # global author_info
                if match_org == None:
                    match_org = org

                    if "year" in o and o["year"] != "":
                        match_year = o["year"]
                elif match_org != org: #有多个简称匹配
                    print('abbrev org error!', author_name, author_list)

                    final_o = match_org_by_score(author_name,author_list)
                    # print("final_o:",final_o)

                    if "org" in final_o and final_o["org"] != "":
                        match_org = final_o["org"]

                        if "year" in final_o and final_o["year"] != "":
                            match_year = final_o["year"]
                        # print("final_org:",match_org)
                        if final_o["name"] == 'Luo Jiayan' and author_name == 'jie_luo':
                            match_org = None
                            match_year = None

            # if o["name"] =="Jun Chen \u9648\u4fca":
            #     print("Jun Chen \u9648\u4fca", match_org)


    if match_flag == False and org_all_null:
        match_flag = True

    return match_org, match_year, match_flag
'''



def getAffiliation(fn_atuhor,fn_paper,all_fn_atuhor,all_fn_paper,saveFile):
    #org为空字符串
    global author_info
    with open(fn_atuhor,"r") as fa, open(fn_paper,"r") as fp ,open(all_fn_atuhor,"r") as all_fa,open(all_fn_paper,"r") as all_fp:
        author_json = json.load(fa)
        paper_json = json.load(fp)

        all_author_json = json.load(all_fa)
        all_paper_json = json.load(all_fp)



        # print(author_json["li_guo"])

        for (key, value) in author_json.items():
            author_name = key #li_guo
            # print(author_name)
            # print(value)
            for authorId,paperIds in value.items():
                author_info.setdefault((authorId, author_name), set())
                # print(authorId)
                for paperId in paperIds:
                    # print(paperId)
                    paper = paper_json[paperId]
                    author_list = paper["authors"]
                    author_list = get_author_set(author_list) #去重
                    match_num = 0

                    tmp_org = set()
                    # print(author_name,paperId)
                    for o in author_list : #Li Guo
                        if o["name"].find(" ") == -1 and o["name"].find("\u00a0") == -1: #Lin-Qi Jianmin\u00a0Zhao == jianmin_zhao
                            continue
                        if aligh_author_name( o["name"] ) == author_name or get_converse_name( o["name"] ) == author_name:
                            match_num = match_num + 1
                            # print("aligh_author_name:",aligh_author_name( o["name"] ),get_converse_name( o["name"] ))
                            if "org" in o :
                                org = o["org"]
                                if org != "":

                                    # match_num = match_num + 1
                                    org = org.replace("\n",'\\n').replace("\t"," ")
                                    org = char_filter(org) #lower #"." -> 导致空
                                # global author_info
                                    if org != "":
                                        tmp_org.add(org)
                    if match_num >1 :

                        print("match overflow:",author_name,paperId)
                    elif match_num == 0:

                        abbrev_match_org, abbrev_match_flag = get_abbrevname_match_org(author_name, author_list)
                        if abbrev_match_org != None:
                            fil_org = char_filter(abbrev_match_org)
                            if fil_org != "":
                                tmp_org.add( fil_org )

                        else :
                            sub_match_org , sub_match_flag = get_substr_org(author_name, author_list)
                            if sub_match_org != None:

                                fil_org = char_filter(sub_match_org)
                                if fil_org != "":
                                    tmp_org.add( fil_org )
                            else:
                                if abbrev_match_flag == False and sub_match_flag ==False :
                                    print("none match :", author_name, paperId)

                        author_info[(authorId, author_name)] = author_info[(authorId, author_name)] | tmp_org


                    else:
                        author_info[(authorId, author_name)] = author_info[(authorId, author_name)] | tmp_org #并集

        #从whole_author中获取机构信息
        for (key, value) in all_author_json.items():
            authorId = key #li_guo
            info = value
            # print(author_name)
            # print(value)

                # authornum += 1
                # print(authorId)
            author_name = info["name"]
            author_info.setdefault((authorId, author_name), set())
            paperIds = info["papers"]
            for paperId in paperIds:
                # print(paperId)
                paper = all_paper_json[paperId]
                author_list = paper["authors"]
                author_list = get_author_set(author_list)
                match_num = 0

                tmp_org = set()
                for o in author_list:  # Li Guo
                    if o["name"].find(" ") == -1 and o["name"].find(
                            "\u00a0") == -1:  # Lin-Qi Jianmin\u00a0Zhao == jianmin_zhao
                        continue
                    if aligh_author_name(o["name"]) == author_name or get_converse_name(o["name"]) == author_name:
                        match_num = match_num + 1
                        # print("aligh_author_name:",aligh_author_name( o["name"] ),get_converse_name( o["name"] ))
                        if "org" in o:
                            org = o["org"]
                            if org != "":
                                org = org.replace("\n", '\\n').replace("\t"," ")
                                # global author_info
                                org = char_filter(org)
                                if org != "":
                                    tmp_org.add(org)
                if match_num > 1:

                    print("match overflow:", author_name, paperId)
                elif match_num == 0:
                    #采用子串匹配

                    abbrev_match_org ,abbrev_match_flag= get_abbrevname_match_org(author_name, author_list)
                    if abbrev_match_org != None:
                        fil_org = char_filter(abbrev_match_org)
                        if fil_org != "":
                            tmp_org.add(fil_org)
                        # tmp_org.add( char_filter( abbrev_match_org ) )

                        # if abbrev_match_org == 'college of civil & transportation engineering, hohai university, nanjing, 210098, china':
                        #     print("target author_info:", author_info[('F2op4LGI', 'jun_chen')])
                        #     print(tmp_org)


                    else:
                        sub_match_org , sub_match_flag = get_substr_org(author_name, author_list)
                        if sub_match_org != None:
                            fil_org = char_filter(sub_match_org)
                            if fil_org != "":
                                tmp_org.add(fil_org)
                            # tmp_org.add( char_filter(sub_match_org) )
                        else:
                            if abbrev_match_flag == False and sub_match_flag == False:
                                print("none match :", author_name, paperId)

                    author_info[(authorId, author_name)] = author_info[(authorId, author_name)] | tmp_org

                else: #match_num == 0:
                    author_info[(authorId, author_name)] = author_info[(authorId, author_name)] | tmp_org  # 并集


        with open(saveFile,"w",encoding="utf-8") as fout:
            authornum = 0
            for (authorId,author_name),orgList in author_info.items():
                authornum += 1
                fout.write(authorId+"\t"+author_name+"\t"+"|||".join(orgList))
                fout.write('\n')

        # print("target author_info:",author_info[('F2op4LGI', 'jun_chen')])



def get_match_org_year(paper, paperId,author_name):

    #year 不合法 返回（org, None）
    author_list = paper["authors"]
    author_list = get_author_set(author_list)  # 去重
    match_num = 0

    # tmp_org = set()
    target_org_year_pair = None
    # print(author_name,paperId)
    paper_year = None

    if "year" in paper and paper["year"] != ""  and paper["year"] != 0  and paper["year"] != 210:
        paper_year = paper["year"]

        if paper_year < 1960:
            print("year exceed:", paper_year)

    for o in author_list:  # Li Guo
        if o["name"].find(" ") == -1 and o["name"].find(
                "\u00a0") == -1:  # Lin-Qi Jianmin\u00a0Zhao == jianmin_zhao
            # print("不包含空格的作者名：", o["name"])
            continue
        if aligh_author_name(o["name"]) == author_name or get_converse_name(o["name"]) == author_name:
            match_num = match_num + 1
            # print("aligh_author_name:",aligh_author_name( o["name"] ),get_converse_name( o["name"] ))
            if "org" in o and o["org"] != "":
                org = o["org"]

                org = org.replace("\n", '\\n').replace("\t", " ")
                org = char_filter(org)  # lower #"." -> 导致空

                if org != "":




                    # if now_year != None :

                    target_org_year_pair = (org, paper_year)

                    # match_num = match_num + 1

    if match_num > 1:

        print("match overflow:", author_name, paperId)
        return None
    elif match_num == 0:

        target_org_year_pair = None

        abbrev_match_org,  abbrev_match_flag = get_abbrevname_match_org(author_name, author_list)
        if abbrev_match_org != None:
            fil_org = char_filter(abbrev_match_org)
            if fil_org != "":


                target_org_year_pair = (fil_org, paper_year)

        else:
            sub_match_org,  sub_match_flag = get_substr_org(author_name, author_list)
            if sub_match_org != None:

                fil_org = char_filter(sub_match_org)
                if fil_org != "":

                    target_org_year_pair = (fil_org, paper_year)
            else:
                if abbrev_match_flag == False and sub_match_flag == False:
                    print("none match :", author_name, paperId)
                    return None

        # if target_org_year_pair != None:
        #     tgt_org, tgt_year = target_org_year_pair
        #     author_info_year.setdefault(id, {}).setdefault(author_name, {}).setdefault(tgt_year,
        #                                                                                set()).add(tgt_org)


    else:  # match_num ==1 1:
        # tgt_org, tgt_year = target_org_year_pair
        # author_info_year.setdefault(id, {}).setdefault(author_name, {}).setdefault(tgt_year,
        #                                                                            set()).add(tgt_org)
        pass

    return target_org_year_pair


def get_match_affiliation(paper, paperId,author_name): #paperId用于打印信息

    #year 不合法 返回（org, None）
    author_list = paper["authors"]
    author_list = get_author_set(author_list)  # 去重
    match_num = 0

    # tmp_org = set()
    target_org = None



    for o in author_list:  # Li Guo
        if o["name"].find(" ") == -1 and o["name"].find(
                "\u00a0") == -1:  # Lin-Qi Jianmin\u00a0Zhao == jianmin_zhao
            # print("不包含空格的作者名：", o["name"])
            continue
        if aligh_author_name(o["name"]) == author_name or get_converse_name(o["name"]) == author_name:
            match_num = match_num + 1
            # print("aligh_author_name:",aligh_author_name( o["name"] ),get_converse_name( o["name"] ))
            if "org" in o and o["org"] != "":
                org = o["org"]

                org = org.replace("\n", '\\n').replace("\t", " ")
                org = char_filter(org)  # lower #"." -> 导致空

                if org != "":

                    target_org = org

    if match_num > 1:

        print("match overflow:", author_name, paperId)
        return None
    elif match_num == 0:

        target_org = None

        abbrev_match_org,  abbrev_match_flag = get_abbrevname_match_org(author_name, author_list)
        if abbrev_match_org != None:
            fil_org = char_filter(abbrev_match_org)
            if fil_org != "":


                target_org = fil_org

        else:
            sub_match_org,  sub_match_flag = get_substr_org(author_name, author_list)
            if sub_match_org != None:

                fil_org = char_filter(sub_match_org)
                if fil_org != "":

                    target_org = fil_org
            else:
                if abbrev_match_flag == False and sub_match_flag == False:
                    print("none match :", author_name, paperId)
                    return None


    else:  # match_num ==1 :
               pass

    return target_org



def get_year_affiliation(fn_atuhor,fn_paper,all_fn_atuhor,all_fn_paper,saveFile="../data/all_author_info_with_year.pickle"):
    author_info_year = dict() #author_info_year.setdefault(id, {}).setdefault(name, {}).setdefault(year , set() )
    with open(fn_atuhor, "r") as fa, open(fn_paper, "r") as fp, open(all_fn_atuhor, "r") as all_fa, open(all_fn_paper,
                                                                                                         "r") as all_fp:
        author_json = json.load(fa)
        paper_json = json.load(fp)

        all_author_json = json.load(all_fa)
        all_paper_json = json.load(all_fp)

        # print(author_json["li_guo"])

        for (key, value) in author_json.items():
            author_name = key  # li_guo
            # print(author_name)
            # print(value)
            for authorId, paperIds in value.items():
                # author_info_year.setdefault(id, {}).setdefault(author_name, [])
                # print(authorId)
                for paperId in paperIds:
                    # print(paperId)
                    paper = paper_json[paperId]

                    target_org_year_pair = get_match_org_year(paper, paperId, author_name)

                    if target_org_year_pair != None:
                        tgt_org, tgt_year = target_org_year_pair
                        author_info_year.setdefault(authorId, {}).setdefault(author_name, {}).setdefault(tgt_year,
                                                                                                   set()).add(tgt_org)

        # 从whole_author中获取机构信息
        for (key, value) in all_author_json.items():
            authorId = key  # li_guo
            info = value
            # print(author_name)
            # print(value)

            # authornum += 1
            # print(authorId)
            author_name = info["name"]
            paperIds = info["papers"]
            for paperId in paperIds:
                # print(paperId)
                paper = all_paper_json[paperId]
                target_org_year_pair = get_match_org_year(paper, paperId, author_name)

                if target_org_year_pair != None:
                    tgt_org, tgt_year = target_org_year_pair
                    author_info_year.setdefault(authorId, {}).setdefault(author_name, {}).setdefault(tgt_year,
                                                                                               set()).add(tgt_org)

        print("author_info_year:",author_info_year)
        pickle.dump(author_info_year, open(saveFile,"wb"))

        save_info = pickle.load(open(saveFile,"rb"))
        print("save_info:",save_info)





def get_affiliation_with_paperId(fn_atuhor,fn_paper,all_fn_atuhor,all_fn_paper,saveFile="../data/all_author_info_with_paperid.pickle"):
    author_info_paperid = dict() #author_info.setdefault(id, {}).setdefault(name, []).extend(org_list)
    with open(fn_atuhor, "r") as fa, open(fn_paper, "r") as fp, open(all_fn_atuhor, "r") as all_fa, open(all_fn_paper,
                                                                                                         "r") as all_fp:
        author_json = json.load(fa)
        paper_json = json.load(fp)

        all_author_json = json.load(all_fa)
        all_paper_json = json.load(all_fp)

        # print(author_json["li_guo"])

        for (key, value) in author_json.items():
            author_name = key  # li_guo
            # print(author_name)
            # print(value)
            for authorId, paperIds in value.items():
                # author_info_paperid.setdefault(id, {}).setdefault(author_name, [])
                # print(authorId)
                for paperId in paperIds:
                    # print(paperId)
                    paper = paper_json[paperId]

                    target_org = get_match_affiliation(paper, paperId, author_name) #paperId用于打印信息

                    if target_org != None:

                        author_info_paperid.setdefault(authorId, {}).setdefault(author_name, {}).setdefault(target_org,set()).add(paperId)

        # 从whole_author中获取机构信息
        for (key, value) in all_author_json.items():
            authorId = key  # li_guo
            info = value
            # print(author_name)
            # print(value)

            # authornum += 1
            # print(authorId)
            author_name = info["name"]
            paperIds = info["papers"]
            for paperId in paperIds:
                # print(paperId)
                paper = all_paper_json[paperId]
                target_org = get_match_affiliation(paper, paperId, author_name)

                if target_org != None:
                    author_info_paperid.setdefault(authorId, {}).setdefault(author_name, {}).setdefault(target_org,set()).add(paperId)

        print("author_info_paperid:",author_info_paperid)
        pickle.dump(author_info_paperid, open(saveFile,"wb"))

        save_info = pickle.load(open(saveFile,"rb"))
        print("save_info:",save_info)


def build_train_pos(fn_author, fout):
    with open(fn_author,"r") as fa ,open(fout,"w",encoding="utf_8") as fo:
        author_json = json.load(fa)

        # print(author_json["li_guo"])
        for (key, value) in author_json.items():
            author_name = key #li_guo
            # print(author_name)
            # print(value)
            for authorId,paperIds in value.items():
                for paperId in paperIds:
                    fo.write("\t".join(("1",authorId,paperId,author_name)))
                    fo.write("\n")




if __name__ == '__main__':
    author_info = dict()
    
    getAffiliation('../data/train_author.json', '../data/train_pub.json', '../data/whole_author_profile.json',
                   '../data/whole_author_profile_pub.json', '../data/all_author_info.txt')

    get_year_affiliation('../data/train_author.json', '../data/train_pub.json', '../data/whole_author_profile.json',
                   '../data/whole_author_profile_pub.json', '../data/all_author_info_with_year.pickle')

    
