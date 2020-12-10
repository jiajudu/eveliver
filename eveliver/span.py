import collections
import re
import string
from typing import List

import nltk
import torch


def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_best_span(a, b):
    # a: T(batch_size, seq_len)
    # b: T(batch_size, seq_len)
    s = a.shape[1]
    ar = torch.arange(s, device=a.device)
    score = a.unsqueeze(1) + b.unsqueeze(2) - (torch.gt(ar.unsqueeze(0) - ar.unsqueeze(1), 0).long() + torch.lt(ar.unsqueeze(0) - ar.unsqueeze(1), -15).long()).unsqueeze(0) * 1e7
    m = score.view(a.shape[0], -1).argmax(1)
    indices = torch.cat(((m % s).view(-1, 1), (m / s).view(-1, 1)), dim=1)
    confidence = torch.exp(torch.max(torch.max(score, dim=1)[0], dim=1)[0] - torch.logsumexp(torch.logsumexp(score, dim=1), dim=1))
    return indices, confidence


def list_find(a, b):
    first_matches = list()
    offset = -1
    element = b[0]
    while True:
        try:
            offset = a.index(element, offset + 1)
        except ValueError:
            break
        first_matches.append(offset)
    ret = list()
    for m in first_matches:
        if m + len(b) <= len(a):
            match = True
            for i in range(1, len(b)):
                if a[m + i] != b[i]:
                    match = False
                    break
            if match:
                ret.append([m, m + len(b) - 1])
    return ret


def findall(a: str, b: str) -> List[int]:
    matches = list()
    offset = 0
    while True:
        o = a.find(b, offset)
        if o >= 0:
            matches.append(o)
            offset = o + 1
        else:
            break
    return matches


def mixed_segmentation(in_str, rm_punc=False):
    in_str = in_str.lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
            '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
            '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)
    return segs_out


def remove_punctuation(in_str):
    in_str = in_str.lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
            '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
            '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = i+1
    return s1[p-mmax:p], mmax


def cqa_f1(answer, prediction):
    if answer == '' and prediction == '':
        return 1
    ans_segs = mixed_segmentation(answer, rm_punc=True)
    prediction_segs = mixed_segmentation(prediction, rm_punc=True)
    lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
    if lcs_len == 0:
        return 0
    precision = 1.0*lcs_len/len(prediction_segs)
    recall = 1.0*lcs_len/len(ans_segs)
    f1 = (2*precision*recall)/(precision+recall)
    return f1


def cqa_em(answer, prediction):
    answer = remove_punctuation(answer)
    prediction_ = remove_punctuation(prediction)
    if answer == prediction_:
        return 1
    return 0
