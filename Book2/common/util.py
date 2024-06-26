import numpy as np
from typing import *

def preprocess(text: str) -> Tuple[np.ndarray, dict, dict]:
    text = text.lower().replace(".", " .")
    words = text.split()
    
    word_to_id = dict()
    id_to_word = dict()
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[word] for word in words])
    return corpus, word_to_id, id_to_word


def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    
    for target_idx in range(window_size, window_size+len(target)):
        context = []
        for context_idx in range(target_idx-window_size, target_idx+window_size+1):
            if context_idx == target_idx:
                continue
            
            context.append(corpus[context_idx])
        
        contexts.append(context)
    
    return np.array(contexts), np.array(target)


def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환

    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot