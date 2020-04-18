from gensim import corpora, models, similarities
from collections import defaultdict
import nltk
import numpy as np
class DOC_SIM:
    def __init__(self,documents):

        self.split_chars = ['\u202f', '\u2212', '\u2013', '\u2218', '\u03bc']
        # 1.分词，去除停用词
        # self.stoplist = set('for a of the and to in'.split())

        self.stoplist = nltk.corpus.stopwords.words('english')
        # print("stoplist:",self.stoplist)

        self.dictionary = None
        self.tfidf_model = None
        self.model_index = None

        self.get_docs_model(documents)

    def unicode_filter(self,str):
        str = str.replace('\u202f',' ')
        str = str.replace('\u2212', '−')
        str = str.replace('\u2013', '-') #–
        str = str.replace('\u2218', '.')#∘
        str = str.replace('\u00a9','©')#©
        # str = str.replace('\u03bc', ' ')#μ

        return str

    def replace_split_chars(self, doc):
        doc = self.unicode_filter(doc)
        if doc[-1] == '.':
            doc = doc[:-1]
        return doc.replace(', ', ' ').replace('. ', ' ')


    def get_docs_model(self,documents):



        documents = list(map(self.replace_split_chars, documents))

        # print("documents:", documents)


        texts = [[word for word in document.lower().split() if word not in self.stoplist] for document in documents]

        # 2.计算词频
        frequency = defaultdict(int)  # 构建一个字典对象
        # 遍历分词后的结果集，计算每个词出现的频率
        for text in texts:
            for token in text:
                frequency[token] += 1
        # 选择频率大于1的词
        texts = [[token for token in text if frequency[token] > 1] for text in texts]
        # print(texts)


        # 3.创建字典（单词与编号之间的映射）
        dictionary = corpora.Dictionary(texts)

        # print(dictionary.token2id)
        # {'human': 0, 'interface': 1, 'computer': 2, 'survey': 3, 'user': 4, 'system': 5, 'response': 6, 'time': 7, 'eps': 8, 'trees': 9, 'graph': 10, 'minors': 11}



        # 5.建立语料库# 将每一篇文档转换为向量
        corpus = [dictionary.doc2bow(text) for text in texts]
        # [[[(0, 1), (1, 1), (2, 1)], [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)], [(1, 1), (4, 1), (5, 1), (8, 1)], [(0, 1), (5, 2), (8, 1)], [(4, 1), (6, 1), (7, 1)], [(9, 1)], [(9, 1), (10, 1)], [(9, 1), (10, 1), (11, 1)], [(3, 1), (10, 1), (11, 1)]]

        # 6.初始化模型
        # 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）
        tfidf_model = models.TfidfModel(corpus)

        # 将整个语料库转为tfidf表示方法
        corpus_tfidf = tfidf_model[corpus]

        # 7.创建索引


        model_index = similarities.MatrixSimilarity(corpus_tfidf)

        self.dictionary = dictionary
        self.tfidf_model = tfidf_model
        self.model_index = model_index




    def get_max_similarity(self,new_doc):
        new_doc = self.replace_split_chars(new_doc)
        new_vec = self.dictionary.doc2bow(new_doc.lower().split())
        new_vec_tfidf = self.tfidf_model[new_vec]  # 将要比较文档转换为tfidf表示方法
        sims = self.model_index[new_vec_tfidf]
        sims = sorted(sims, reverse=True)
        # print(sims ) #[0.30510765 0.07911602 0.27231705 0.07911602 0.08711167 0.04326698 1.0000001 ]
        return np.average(sims[1:] )#防止样本中的论文为作者论文档案之一 存在偏差



if __name__ == "__main__":

    documents = [
        "A completely flexible fiber-shaped dye sensitized solar cell has been truly realized for the first time, due to a novel photoanode with a TiO2 micron-cone-nanowire array structure prepared by a simple two-step process. The TiO2 micron-cone array, made using an electrochemical method, is used as a frame for a novel photoanode, with its roots sinking deep down into a Ti wire substrate. The TiO2 nanowire array is coated onto the TiO2 micron-cone surface by a hydrothermal reaction to form a dye-adsorption layer to the enhance power conversion efficiency of the novel device. With a high dye-adsorption capacity and strong combination between the TiO2 micron-cone and Ti substrate, the minimum bending radius of the photoanode could reach 0.45 mm, and 96.6% retention of the initial conversion efficiency was obtained after bending 100 times.",
        "The performances of fiber dye-sensitized solar cells (FDSSCs), in terms of charge collection efficiency, light-harvesting ability, and structural stability, are improved through a novel hierarchically structured photoanode based on a Ti microridge/nanorod-modified wire substrate. The microridge made of several Ti micropits is inserted into TiO2 layer to shorten the photoelectron transport distance from the original place in the TiO2 layer to substrate and to increase the electron transport rate. The Ti micropits are used as light-gathering centers to collect the reflected light. Meanwhile, the Ti nanorods are evenly distributed on the surface of the microridge-coated Ti wire substrate, which increases the contact area between the substrate and the TiO2 layer in order to suppress the electron recombination and scatters the incident light to further improve the light-harvesting ability. Therefore, the charge collection and power conversion efficiencies of the novel FDSSC have been accordingly enhanced by 17.7% and 61.6%, respectively, compared with traditional FDSSC. Moreover, the structural stability of the novel FDSSC has been strengthened.",
        "Abstract   A spring-like Ti@TiO 2  nanowire array wire has been introduced into a stretchable fiber-shaped dye-sensitized solar cell (FDSSC) as a photoanode to achieve high flexibility and elasticity in this paper. Given the TiO 2  layer, which was prepared by a hydrothermal reaction in the optimized NaOH concentration of 2.5\u202fmol\u202fL \u22121 , with a 1D structure and high adhesion between the TiO 2  nanowire array and the Ti wire substrate, the novel FDSSC still possesses photoelectric conversion efficiency retention rates of approximately 97.00% and 95.95% after bending to a radius of 1.0\u202fcm and stretching to 100% strain, respectively. EIS result shows the degradation mechanism of the FDSSC photoelectric performance: the bending test leads to more terrible electron combination; the stretching operation increases the internal resistance and charge-transfer resistance at the counter electrode. Moreover, itu0027s worth noting that, this is the first time to show a 100%-stretching degree in the FDSSC research field so far.",
        "The performances of fiber dye-sensitized solar cells (FDSSCs), in terms of charge collection efficiency, light-harvesting ability, and structural stability, are improved through a novel hierarchically structured photoanode based on a Ti microridge/nanorod-modified wire substrate. The microridge made of several Ti micropits is inserted into TiO2 layer to shorten the photoelectron transport distance from the original place in the TiO2 layer to substrate and to increase the electron transport rate. The Ti micropits are used as light-gathering centers to collect the reflected light. Meanwhile, the Ti nanorods are evenly distributed on the surface of the microridge-coated Ti wire substrate, which increases the contact area between the substrate and the TiO2 layer in order to suppress the electron recombination and scatters the incident light to further improve the light-harvesting ability. Therefore, the charge collection and power conversion efficiencies of the novel FDSSC have been accordingly enhanced by 17.7% and 61.6%, respectively, compared with traditional FDSSC. Moreover, the structural stability of the novel FDSSC has been strengthened.",
        "A conformal and thin TiO2 film fabricated with atomic layer deposition (ALD) improves the performance of fiber-shaped dye-sensitized solar cells (FDSSCs). Electrical contact at Ti/TiO2 is improved through insertion of a uniform and pinhole-free TiO2 film. The film thickness varies from 5 nm to 40 nm, and the high power conversion efficiency of 7.41% is achieved from a typical device with a 15 nm TiO2 film and a 10 \u03bcm TiO2 nanoparticles layer. The compact and high-quality TiO2 film improves Ti/TiO2 interfacial property. Surface modification hastens electron transport, which thus improves the charge collection efficiency. Both photocurrent density and bending performance are significantly enhanced compared with those of the corresponding FDSSCs without ALD treatment.",
        "Nanoparticles (NPs) with high uniformity have been extensively investigated for their excellent chemical stability. Near-monodisperse globular MoS2 NPs were prepared with sulphur powders (SPs) as a sulphur source by a one-pot polyol-mediated process without surfactants, transfer agents and toxic agents at 170\u2013190\u2218C. The as-processed SPs greatly affected the formation of the MoS2 NPs after low-activity sulphur (S8)n was reassembled from common SPs (S8). The average size of MoS2 NPs can be reduced remarkably from 100\u2013200nm to 50nm by introducing low amounts of MnCl2. A preliminary four-step growth mechanism based on the aggregation-coalescence model was also proposed. This green and simple method may be an alternative to the common hot-injection and heating-up methods for the preparation of monodisperse NPs, particularly transition metal dichalogenides.",
        "Ti/TiO2 micron-cone array (TMCA) prepared through a synergy of chemical and electrochemical reactions was used as the frame and electron transfer channel for the photoanode of a fiber dye-sensitized solar cell (FDSSC). A TiO2 multilayer structure composed of compact, light scattering, and porous layers was introduced into the fiber photoanode for the first time based on the TMCA. The novel fiber photoelectric device attained a photoelectric conversion efficiency of 8.07%, the highest value for FDSSCs with a Pt wire counter electrode to date, and a good flexibility."
    ]

    # documents = [
    #     "A completely flexible fiber-shaped dye sensitized solar cell has been truly realized for the first time, due to a novel photoanode with a TiO2 micron-cone-nanowire array structure prepared by a simple two-step process. The TiO2 micron-cone array, made using an electrochemical method, is used as a frame for a novel photoanode, with its roots sinking deep down into a Ti wire substrate. The TiO2 nanowire array is coated onto the TiO2 micron-cone surface by a hydrothermal reaction to form a dye-adsorption layer to the enhance power conversion efficiency of the novel device. With a high dye-adsorption capacity and strong combination between the TiO2 micron-cone and Ti substrate, the minimum bending radius of the photoanode could reach 0.45 mm, and 96.6% retention of the initial conversion efficiency was obtained after bending 100 times.",
    #
    # ]
    # documents = ['a wet procedure to prepare stoichiometric and homogeneous zr-rich pzt powders is described. the hydroxide precursor was prepared by coprecipitation from a mixed nitrate solution containing pb', 'a wet procedure to prepare stoichiometric and homogeneous zrdrich pzt powders was described. the hydroxide precursor was prepared by coprecipitation from a mixed nitrate solution containing pb']
    # ds = None
    # try:
    #     ds = DOC_SIM(documents)
    # except ValueError:
    #     pass

    ds = DOC_SIM(documents)
    # dictionary, tfidf_model, model_index = ds.get_docs_model(documents)
    new_doc = "Ti/TiO2 micron-cone array (TMCA) prepared through a synergy of chemical and electrochemical reactions was used as the frame and electron transfer channel for the photoanode of a fiber dye-sensitized solar cell (FDSSC). A TiO2 multilayer structure composed of compact, light scattering, and porous layers was introduced into the fiber photoanode for the first time based on the TMCA. The novel fiber photoelectric device attained a photoelectric conversion efficiency of 8.07%, the highest value for FDSSCs with a Pt wire counter electrode to date, and a good flexibility."

    print( ds.get_max_similarity(new_doc) )
    new_doc1 = "Objective To find an efficient way for developing olfactory neural chip and its measurement system of electro-physiological signals with Multi-electrode Array(MEA),furthermore,to verify the efficiency of this detection system for electro-physiological signals of neural system and others.Methods The acquisitions from electro-cardio signals of rat hearts in vitro and from self-issuing signals of hippocampus neural networks on MEA chip,as well the electro-physiological signals of olfactory bulb neural network neuron chips made of cultured neonatal rat olfactory bulb neuron according to different shapes at different culturing periods,were performed with 72-channel signal condition circuits and an acquisition system developed in our lab.Those shapes of olfactory bulb networks at different culturing periods were observed and compared by optical microscopy.Results It was apparent that the detection platform integrated with MEA sensors could be used to capture electro-physiological signals of olfactory bulb neural networks at different cultured periods,and those data acquired could be used to analyze complexity and temporal coding of electrophysiological signals of neural networks.Conclusion The neural signal detection platform developed here is capable of being broadly used on the studies of neural electrophysiology and neural biology."
    print(ds.get_max_similarity(new_doc1))
