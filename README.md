# OAG-WhoIsWho 赛道二 




在由AMiner、智源举办的同名消岐竞赛中取得0.95097的成绩，在Final Leaderboard中排名第5，获取奖金￥2,000。

[比赛链接][1]

[证书][2]




## 简介 ##
由于论文分配算法的局限性，现有的收录各种论文的学术系统内部存在着大量的论文分配错误。赛题要求根据论文的详细信息以及作者与论文之间的联系，去区分属于不同作者的同名论文，然后准确快速的将论文分配到系统中已有作者档案，获得良好的论文消歧结果。



## 预处理 ##

 - 数据清洗：词形还原、去停用词、去特殊字符、转小写
 - 由于机构名是作者消岐的强特征，预先为每个作者建立机构库
 - 根据论文标题、摘要为bert构建训练语料
 - 将同名作者作为负样本构建训练数据。为了提升准确率且避免过拟合，负样本选择发表论文数最多的top5，再从剩余样本中随机挑选3个加入负样本



## 特征工程 ##

 1. 机构类
 
    机构间的编辑距离、相似度、jaccard
    
    当前paper其余作者机构 与当前作者机构的相似度信息
    

 2. 协作者类
    
    共同合作者占所有作者的比率
    
    有相同合作者的论文比率
    
    当前论文的作者与当前作者所有论文的作者有交集的最多作者数
    
    当前论文的作者与当前作者所有论文的作者有交集的论文数目
    
    ......
    

 3. 标题、关键词类
 
       标题、关键词间的编辑距离、相似度、jaccard
 
       基于word2vec、triplet loss训练embedding计算相似度

    
  

 4. 摘要等短文本类
    最大tf-idf相似度
    
    当前论文的摘要，与候选作者文本语料经过bert模型衡量相似度特征

 5. 会议名称类
 
    venue的jaccard
    
 6. 发表年份类
 
    时间跨度



## 集成学习 ##

 - 调参
 
   以交叉验证的auc作为评价指标
   
   网格搜索与手动调节相结合
   
 - 集成
 
   集成了xgboost、lightGBM、catboost三个梯度提升模型
   
   投票法 + 均值法 + stacking



## 运行步骤 ##
1. 将数据文件train_pub.json 、train_author.json 、whole_author_profile.json、whole_author_profile_pub.json、cna_valid_unass_competition.json、cna_valid_pub.json、cna_test_unass_competition.json、cna_test_pub.json放置于NDModel\data\下

2. 在NDModel\util目录下执行data_process.py，会在data目录下生成 all_author_info.txt、  all_author_info_with_year.pickle、 all_author_info_with_paperid.pickle文件，这三个文件为预处理的作者的档案信息；执行get_corpus.py，获得训练embedding以及bert的语料

3. 执行train_triplet.py，训练短文本的embedding

4. 执行gen_feature.py，会在NDModel\data\feature目录下生成feature_list_train.txt，在NDModel\data\instance目录下生成train_instance.pickle
5. 执行model.py，进行训练与预测，在NDModel\model目录下生成xgboost_best.model（模型）、paper2candidates.pkl、paper2features.pkl（保存预测中间变量），在NDModel\data\res目录下生成result.json（最终结果文件）

    

    
   





 


  [1]: https://www.biendata.com/competition/aminer2019_2/
  [2]: https://drive.google.com/open?id=1SY3f6JXr-xZw_LknfwfyD5NTOA_hLsga
