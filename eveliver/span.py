import collections
import re
import string

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
