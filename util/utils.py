import json
import pickle
from pyjarowinkler import distance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer #词型还原
from nltk.corpus import stopwords
import re
import string


class TextToVec:
    def __init__(self):
        super().__init__()
        self.my_stopwords = set(stopwords.words('english'))
        self.num_pattern = re.compile(r'\d+')
        self.remove_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_dict = load_pickle('../data/glove.word2vec.dict.pkl')

    def clean_text(self, str_info):
        str_lower = str_info.lower().strip()
        result = str_lower.translate(self.remove_punctuation)
        result = self.num_pattern.sub('', result)
        tokens = word_tokenize(result)
        result = [word for word in tokens if word not in self.my_stopwords]
        result = [self.lemmatizer.lemmatize(word) for word in result]
        # print(result)
        return result

    def get_vec(self, str_info):
        # return np.array("0.27204 -0.06203 -0.1884 0.023225 -0.018158 0.0067192 -0.13877 0.17708 0.17709 2.5882 -0.35179 -0.17312 0.43285 -0.10708 0.15006 -0.19982 -0.19093 1.1871 -0.16207 -0.23538 0.003664 -0.19156 -0.085662 0.039199 -0.066449 -0.04209 -0.19122 0.011679 -0.37138 0.21886 0.0011423 0.4319 -0.14205 0.38059 0.30654 0.020167 -0.18316 -0.0065186 -0.0080549 -0.12063 0.027507 0.29839 -0.22896 -0.22882 0.14671 -0.076301 -0.1268 -0.0066651 -0.052795 0.14258 0.1561 0.05551 -0.16149 0.09629 -0.076533 -0.049971 -0.010195 -0.047641 -0.16679 -0.2394 0.0050141 -0.049175 0.013338 0.41923 -0.10104 0.015111 -0.077706 -0.13471 0.119 0.10802 0.21061 -0.051904 0.18527 0.17856 0.041293 -0.014385 -0.082567 -0.035483 -0.076173 -0.045367 0.089281 0.33672 -0.22099 -0.0067275 0.23983 -0.23147 -0.88592 0.091297 -0.012123 0.013233 -0.25799 -0.02972 0.016754 0.01369 0.32377 0.039546 0.042114 -0.088243 0.30318 0.087747 0.16346 -0.40485 -0.043845 -0.040697 0.20936 -0.77795 0.2997 0.2334 0.14891 -0.39037 -0.053086 0.062922 0.065663 -0.13906 0.094193 0.10344 -0.2797 0.28905 -0.32161 0.020687 0.063254 -0.23257 -0.4352 -0.017049 -0.32744 -0.047064 -0.075149 -0.18788 -0.015017 0.029342 -0.3527 -0.044278 -0.13507 -0.11644 -0.1043 0.1392 0.0039199 0.37603 0.067217 -0.37992 -1.1241 -0.057357 -0.16826 0.03941 0.2604 -0.023866 0.17963 0.13553 0.2139 0.052633 -0.25033 -0.11307 0.22234 0.066597 -0.11161 0.062438 -0.27972 0.19878 -0.36262 -1.0006e-05 -0.17262 0.29166 -0.15723 0.054295 0.06101 -0.39165 0.2766 0.057816 0.39709 0.025229 0.24672 -0.08905 0.15683 -0.2096 -0.22196 0.052394 -0.01136 0.050417 -0.14023 -0.042825 -0.031931 -0.21336 -0.20402 -0.23272 0.07449 0.088202 -0.11063 -0.33526 -0.014028 -0.29429 -0.086911 -0.1321 -0.43616 0.20513 0.0079362 0.48505 0.064237 0.14261 -0.43711 0.12783 -0.13111 0.24673 -0.27496 0.15896 0.43314 0.090286 0.24662 0.066463 -0.20099 0.1101 0.03644 0.17359 -0.15689 -0.086328 -0.17316 0.36975 -0.40317 -0.064814 -0.034166 -0.013773 0.062854 -0.17183 -0.12366 -0.034663 -0.22793 -0.23172 0.239 0.27473 0.15332 0.10661 -0.060982 -0.024805 -0.13478 0.17932 -0.37374 -0.02893 -0.11142 -0.08389 -0.055932 0.068039 -0.10783 0.1465 0.094617 -0.084554 0.067429 -0.3291 0.034082 -0.16747 -0.25997 -0.22917 0.020159 -0.02758 0.16136 -0.18538 0.037665 0.57603 0.20684 0.27941 0.16477 -0.018769 0.12062 0.069648 0.059022 -0.23154 0.24095 -0.3471 0.04854 -0.056502 0.41566 -0.43194 0.4823 -0.051759 -0.27285 -0.25893 0.16555 -0.1831 -0.06734 0.42457 0.010346 0.14237 0.25939 0.17123 -0.13821 -0.066846 0.015981 -0.30193 0.043579 -0.043102 0.35025 -0.19681 -0.4281 0.16899 0.22511 -0.28557 -0.1028 -0.018168 0.11407 0.13015 -0.18317 0.1323".split(),dtype=np.float)

        result = self.clean_text(str_info)
        data = []
        for word in result:
            data.append(self.word2vec_dict.get(word, np.zeros(300)).tolist())
        if len(data) == 0:
            data = np.zeros(300)
        else:
            data = np.mean(np.array(data), axis=0)
        return data


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", "").replace("-", " ").replace("_", ' ').split()]
    full_name = '_'.join(x)
    return full_name

def get_name_index(target_name, name_list):
    scores = []
    for name in name_list:
        if name == '':
            scores.append(0)
            continue
        score = distance.get_jaro_distance(target_name, name, winkler=True, scaling=0.1)
        target_component = set(target_name.split('_'))
        name_component = set(name.split('_'))
        add_score = len(target_component & name_component) / len(target_component | name_component)
        score = score + add_score
        scores.append(score)
    # print('-'*50)
    # index = np.argsort(scores)
    # print(target_name)
    # print(np.array(name_list)[index][-5:])
    return np.argmax(scores)

