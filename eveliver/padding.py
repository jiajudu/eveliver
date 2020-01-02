import torch


def sents2t(sentences, seq_len, default=0):
    ret = torch.ones(len(sentences), seq_len, dtype=torch.int64) * default
    for _id, sentence in enumerate(sentences):
        ret[_id, 0:len(sentence)] = torch.tensor(sentence, dtype=torch.int64)
    return ret


def b_sents2t(batch, seq_len, default=0):
    ret = torch.ones(len(batch), len(batch[0]), seq_len, dtype=torch.int64) * default
    for bid, sentences in enumerate(batch):
        for sid, sentence in enumerate(sentences):
            ret[bid, sid, 0:len(sentence)] = torch.tensor(sentence, dtype=torch.int64)
    return ret
