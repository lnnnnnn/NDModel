import random, pickle, json
import numpy as np
import sys
sys.path.append("util")
from collections import defaultdict

from sklearn.svm import SVC
sys.path.append("gen_feature")

# from gen_feature.doc_similarity import DOC_SIM
from gen_feature1 import DataCollection

import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score as CVS
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

def grid_search():

    parameters = {
        'max_depth': [5, 10, 15, 20, 25],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
        'n_estimators': [500, 1000, 2000, 3000, 5000],
        'min_child_weight': [0, 2, 5, 10, 20],
        'max_delta_step': [0, 0.2, 0.6, 1, 2],
        'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
        'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]

    }

    model= XGBClassifier(max_depth=10,
            learning_rate=0.01,
            n_estimators=2000,
            silent=True,
            objective='rank:pairwise', #二分类逻辑回归，输出为概率
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=0.85,
            colsample_bytree=0.7,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            seed=1440,
            missing=None)
    gsearch = GridSearchCV(model, param_grid=parameters, scoring='accuracy', cv=3)
    gsearch.fit(x_train, y_train)

    best_model = gsearch.best_estimator_

    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def ensemble_vote():
    model_xg = pickle.load(open('model/model_xg.model', 'rb'))
    model_lgb = pickle.load(open('model/model_lgb.model', 'rb'))
    model_cat = pickle.load(open('model/model_cat.model', 'rb'))

    print("==================================================")
    print("ensemble predicting ...")

    paper2candidates = pickle.load(open('model/paper2candidates.pkl', "rb"))
    paper2features = pickle.load(open('model/paper2features.pkl', "rb"))

    assert len(paper2candidates) == len(paper2features)
    print(len(paper2candidates))
    result_dict = defaultdict(list)

    for paper_id, ins_feature_list in paper2features.items():
        score_list = []
        score_list1 = []
        score_list2 = []
        mp = defaultdict(int)
        for ins in ins_feature_list:
            # 对一篇paper的所有candidate author去打分，利用分数进行排序，取top-1 author作为预测的author
            prob_pred = model_xg.predict_proba([ins])[:, 1]
            prob_pred1 = model_lgb.predict_proba([ins])[:, 1]
            prob_pred2 = model_cat.predict_proba([ins])[:, 1]

            score_list.append(prob_pred[0])
            score_list1.append(prob_pred1[0])
            score_list2.append(prob_pred2[0])

        rank = np.argsort(-np.array(score_list))
        predict_author = paper2candidates[paper_id][rank[0]]
        mp[predict_author] += 1
        rank1 = np.argsort(-np.array(score_list1))
        predict_author1 = paper2candidates[paper_id][rank1[0]]
        mp[predict_author1] += 1

        rank2 = np.argsort(-np.array(score_list2))
        predict_author2 = paper2candidates[paper_id][rank2[0]]
        mp[predict_author2] += 1
        tps = []
        for key, v in mp.items():
            tps.append( (key, v) )
        tps = sorted(tps, key=lambda x:x[1], reverse=True)

        predict_author = tps[0][0] #选出投票数最多的作者id

        result_dict[predict_author].append(paper_id)

    with open("data/res/result_vote.json", 'w') as files:
        json.dump(result_dict, files, indent=4)


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

    # 将x进行标准化
    # std = StandardScaler()
    # x_train = std.fit_transform(x_train)
    # print("standard train:",x_train[:3])

    model_xg = XGBClassifier(objective='rank:pairwise',
                          n_estimators=150,
                          learning_rate=0.05,
                          subsample=0.77,
                          max_depth=7,
                          gamma=2,
                          min_child_weight=2,
                          reg_alpha=0,
                          reg_lambda=1,
                          random_state=420

                          )
    # print(CVS(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()) #-0.012691081051358188
    model_xg.fit(x_train, y_train)
    pickle.dump(model_xg, open('model/model_xg.model', "wb"))



    model_lgb = LGBMClassifier(max_depth=6,
                               learning_rate=0.01,
                               n_estimators=200,
                               objective='binary',  # regression
                               subsample=0.8,
                               num_leaves=50
                               )
    model_lgb.fit(x_train, y_train)

    pickle.dump(model_lgb, open('model/model_lgb.model', "wb"))

    # print(CVS(model_lgb, x_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()) #-0.010733531287803806

    model_cat = CatBoostClassifier(iterations=220,
                                   learning_rate=0.02,
                                   depth=4,
                                   loss_function='Logloss',
                                   eval_metric='Logloss'
                                   )

    print(CVS(model_cat, x_train, y_train, cv=5, scoring='neg_mean_squared_error').mean())  # -0.011547344110854504
    model_cat.fit(x_train, y_train)
    pickle.dump(model_cat, open('model/model_cat.model', "wb"))



    print("==================================================")
    print("training done!")

    model_xg = pickle.load(open('model/model_xg.model','rb'))
    model_lgb = pickle.load(open('model/model_lgb.model','rb'))
    model_cat = pickle.load(open('model/model_cat.model','rb'))


    print("==================================================")
    print("ensemble predicting ...")


    paper2candidates = pickle.load(open('model/paper2candidates.pkl',"rb"))
    paper2features = pickle.load(open('model/paper2features.pkl',"rb"))

    assert len(paper2candidates) == len(paper2features)
    print(len(paper2candidates))
    result_dict = defaultdict(list)

    for paper_id, ins_feature_list in paper2features.items():
        score_list = []
        for ins in ins_feature_list:
            # 对一篇paper的所有candidate author去打分，利用分数进行排序，取top-1 author作为预测的author
            prob_pred = np.array(model_xg.predict_proba([ins])[:, 1]) + np.array(model_lgb.predict_proba([ins])[:, 1]) + np.array(model_cat.predict_proba([ins])[:, 1])
            prob_pred = prob_pred.tolist()
            score_list.append(prob_pred[0])

        rank = np.argsort(-np.array(score_list))
        # 取top-1 author作为预测的author
        predict_author = paper2candidates[paper_id][rank[0]]
        result_dict[predict_author].append(paper_id)

    with open("data/res/result_ensemble.json", 'w') as files:
        json.dump(result_dict, files, indent=4)

#0.9453483383386687
'''
output example:
clf.predict_proba([ins]): [[0.99688712 0.00311288]]
prob_pred: [0.00311288]
clf.predict_proba([ins]): [[0.99245277 0.00754723]]
prob_pred: [0.00754723]
score_list: [0.0011301373194890487, 0.42015922546451334, 0.0430128422479119, 0.00245536462508971, 0.0022167359127426255, 0.0018449978103299023, 0.0017126733127545554, 0.0067398937009484515, 0.012318221342450131, 0.03515801674263342, 0.034505186155567275, 0.012412452818924422, 0.0024719264243943974, 0.20896032446927168, 0.0009324781491365738, 0.0015246844711020698, 0.002208493249495545, 0.004389136935041774, 0.003843878802913376, 0.0013802601417594782, 0.0027455834747019923, 0.0034537641781041727, 0.0016701926292981462, 0.024983380328218483, 0.9936449074947603, 0.0008102835558761402, 0.002504114520475544, 0.0013738919245772834, 0.004695343347437404, 0.0473582033519938, 0.13749251561701448, 0.0008007031156226284, 0.04970868379169312, 0.0012321930485816239, 0.14462569462248004, 0.0015957859763510956, 0.004679046992591728, 0.003968374932952552, 0.03500490491191958, 0.0012980077345307803, 0.12868571746997037, 0.002338612952308998, 0.0017273526027787233, 0.002019618509160497, 0.792544686698087, 0.007462458401703172, 0.001091754528365249, 0.003112876621195221, 0.007547229735836103]
rank: [24 44  1 13 34 30 40 32 29  2  9 38 10 23 11  8 48 45  7 28 36 17 37 18
 21 47 20 26 12  3 41  4 16 43  5 42  6 22 35 15 19 27 39 33  0 46 14 25
 31]

'''