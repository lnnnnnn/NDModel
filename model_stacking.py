import random, pickle, json
import numpy as np
import sys
sys.path.append("util")
from collections import defaultdict

from sklearn.svm import SVC
# from gen_feature.gen_feature1 import DataCollection
sys.path.append("gen_feature")

# from gen_feature.gen_feature import DataCollection
# from gen_feature.doc_similarity import DOC_SIM
from gen_feature1 import DataCollection

import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold #交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.metrics import accuracy_score
import os
from sklearn.preprocessing import StandardScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_train(fn_train_feature):
    global train_ins

    with open(fn_train_feature,"r",encoding="utf-8") as f_triain:
        for line in f_triain:
            feas = line.strip().split(" ")
            feas = list(map(lambda x: float(x),feas))
            train_ins.append((feas[:-1],feas[-1]))
    print(train_ins[:10])




if __name__ == "__main__":
    train_ins = []
    read_train("data/feature/feature_list_train.txt")
    random.shuffle(train_ins)

    x_train = []
    y_train = []
    for ins in train_ins:
        x_train.append(ins[0])
        y_train.append(ins[1])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model_path = "model/xgboost_best.model"

    with open(model_path, "rb") as fm:
        best_model = pickle.load(fm)

    pred_train_raw = best_model.predict(x_train)
    pred_train_raw = np.array(pred_train_raw)
    # for i in range(len(pred_train_raw)):
    #     if pred_train_raw[i] > 0.5:
    #         pred_train_raw[i] = 1
    #     else:
    #         pred_train_raw[i] = 0
    # print(accuracy_score(y_train, pred_train_raw)) #0.9460992907801419

    print("==================================================")
    print("training again ...")
    new_x_train = np.column_stack( (x_train,pred_train_raw) )
    new_y_train = y_train

    print("new_x_train0_len：", len(new_x_train[0]))
    print("new_x_train0：", new_x_train[0])


    new_model = XGBClassifier(objective='rank:pairwise')
    # model = XGBClassifier()
    new_model.fit(new_x_train, new_y_train)

    print("training again done!")
    new_model_path = "model/new_xgboost_best.model"
    pickle.dump(new_model, open(new_model_path, "wb"))

    with open(new_model_path, "rb") as fm:
        new_best_model = pickle.load(fm)


    print("==================================================")
    print("predicting ...")
    # paper2candidates, paper2features = gen_test_instance('data/cna_valid_unass_competition.json', 'data/cna_valid_pub.json',"data/all_author_info.txt",'data/feature/feature_predict.txt',write = True)

    #gen_test_instance(fn_author_predict,fn_paper,fn_author, fn_author_info,fout_feature_test,write=True):
    #gen_test_instance(fn_author_predict,fn_paper,fn_paper_cna,fn_author, fn_author_info,fout_feature_test,write=True):
    # paper2candidates, paper2features = gen_test_instance('data/cna_valid_unass_competition_sample.json', 'data/whole_author_profile_pub.json','data/cna_valid_pub.json','data/whole_author_profile.json',"data/all_author_info.txt",'data/feature/feature_predict.txt',write = True)

    dc = DataCollection(
    fn_author_info = "data/all_author_info.txt",
    fn_author_info_year="data/all_author_info_with_year.pickle",
    # fn_author_info_paperid="data/all_author_info_with_paperid.pickle",
    fn_train_pub = 'data/train_pub.json',
    fn_train_author = "data/train_author.json",
    fn_whole_author = 'data/whole_author_profile.json',
    fn_whole_pub = 'data/whole_author_profile_pub.json',
    fn_valid_pub = "data/cna_test_pub.json")  #data/cna_valid_pub.json

    # paper2candidates, paper2features =dc.gen_test_instance('data/cna_test_unass_competition.json', #data/cna_valid_unass_competition.json
    #                      'data/feature/feature_predict.txt',
    #                                                      write=True)
    # pickle.dump(paper2candidates, open('model/paper2candidates.pkl',"wb"))
    # pickle.dump(paper2features, open('model/paper2features.pkl',"wb"))

    paper2candidates = pickle.load( open('model/new_paper2candidates.pkl',"rb") )
    paper2features = pickle.load( open('model/new_paper2features.pkl',"rb") )






    assert len(paper2candidates) == len(paper2features)
    print(len(paper2candidates))


    # print("paper2features_std:",paper2features[:3])

    result_dict = defaultdict(list)

    index = 0
    for paper_id, ins_feature_list in paper2features.items():
        score_list = []
        for ins in ins_feature_list:
            # 利用svm对一篇paper的所有candidate author去打分，利用分数进行排序，取top-1 author作为预测的author
            prob_pred = best_model.predict_proba([ins])[:, 1]
            ins.append(prob_pred[0])

            new_prob_pred = new_best_model.predict_proba( [ins] )[:, 1]

            score_list.append(new_prob_pred[0])

            if index == 0:
                print("predict ins_len:", len(ins))
                print("predict ins:", ins)

            # print("clf.predict_proba([ins]):",clf.predict_proba([ins]))
            # print("prob_pred:",prob_pred)
        # print("score_list:",score_list)
        rank = np.argsort(-np.array(score_list))
        # print("rank:",rank)
        # 取top-1 author作为预测的author
        predict_author = paper2candidates[paper_id][rank[0]]
        result_dict[predict_author].append(paper_id)

        index += 1

    with open("data/res/result_xg_ensemble.json", 'w') as files:
        json.dump(result_dict, files, indent=4)

#0.9453483383386687
'''
clf.predict_proba([ins]): [[0.99688712 0.00311288]]
prob_pred: [0.00311288]
clf.predict_proba([ins]): [[0.99245277 0.00754723]]
prob_pred: [0.00754723]
score_list: [0.0011301373194890487, 0.42015922546451334, 0.0430128422479119, 0.00245536462508971, 0.0022167359127426255, 0.0018449978103299023, 0.0017126733127545554, 0.0067398937009484515, 0.012318221342450131, 0.03515801674263342, 0.034505186155567275, 0.012412452818924422, 0.0024719264243943974, 0.20896032446927168, 0.0009324781491365738, 0.0015246844711020698, 0.002208493249495545, 0.004389136935041774, 0.003843878802913376, 0.0013802601417594782, 0.0027455834747019923, 0.0034537641781041727, 0.0016701926292981462, 0.024983380328218483, 0.9936449074947603, 0.0008102835558761402, 0.002504114520475544, 0.0013738919245772834, 0.004695343347437404, 0.0473582033519938, 0.13749251561701448, 0.0008007031156226284, 0.04970868379169312, 0.0012321930485816239, 0.14462569462248004, 0.0015957859763510956, 0.004679046992591728, 0.003968374932952552, 0.03500490491191958, 0.0012980077345307803, 0.12868571746997037, 0.002338612952308998, 0.0017273526027787233, 0.002019618509160497, 0.792544686698087, 0.007462458401703172, 0.001091754528365249, 0.003112876621195221, 0.007547229735836103]
rank: [24 44  1 13 34 30 40 32 29  2  9 38 10 23 11  8 48 45  7 28 36 17 37 18
 21 47 20 26 12  3 41  4 16 43  5 42  6 22 35 15 19 27 39 33  0 46 14 25
 31]

'''