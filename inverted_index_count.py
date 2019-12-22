import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter 
from functools import wraps
from time import time

def timeit(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            value = str(end_ if end_ > 0 else 0)
            print("Total execution time: "+ value +" ms")
    return _time_it

@timeit
def get_unique_words_word_list(text_base):
    word_list = {}
    all_unique_words = []
    for index_base, base in enumerate(text_base):
        split_words = base.split()
        word_list[index_base] = split_words
        all_unique_words += split_words        
    all_unique_words = list(set(all_unique_words))
    return word_list,all_unique_words


def avg(x):
    return (sum(x)/len(x))

@timeit
def create_inverted_index_of_ns(all_unique_words,text_var):
    all_time = []
    #converting all the non-standard list to inverted index of unqiue words in standard
    ns_word_list = {}
    for word in all_unique_words:
        ns_word_list[word] = []
        index_ns = 0
        for ns_title in text_var:
            if word in ns_title:
                ns_word_list[word].append(index_ns)
            index_ns += 1
    return ns_word_list

@timeit
def calculate_score_by_count(word_list,ns_word_list):
    score = {}
    for key in word_list:
        score[key] = {}
        for word in word_list[key]:
            ns_indexes = ns_word_list[word]
            for ns_index in ns_indexes:
                if ns_index in score[key]:
                    score[key][ns_index] += 1
                else:
                    score[key][ns_index] = 0 
    return score

@timeit
def get_valid_matches(score):
    valid_score = {}
    for key in score:
        for ns_index in score[key]:
            if score[key][ns_index] > 3:
                if key in valid_score:
                    valid_score[key].append(ns_index)
                else:
                    valid_score[key] = []
    return valid_score


def find_matches(text_base,text_var):
    print("Getting unique words")
    word_list, all_unique_words = get_unique_words_word_list(text_base)
    print("Getting inverted index")
    ns_word_list = create_inverted_index_of_ns(all_unique_words,text_var)
    print("Getting score")
    score = calculate_score_by_count(word_list,ns_word_list)
    print("Getting valid score")
    valid_score = get_valid_matches(score)
    return valid_score


if __name__ == "__main__":
    data_ns = pd.read_csv("non_standard_titles.csv")
    data_base = pd.read_csv("standard_titles.csv")
    text_base = data_base['product_title'].tolist()
    text_var = data_ns["Title"].tolist()
    valid_score = find_matches(text_base,text_var)
    #print(valid_score)
