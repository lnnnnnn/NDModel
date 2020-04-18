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
