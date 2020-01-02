import torch


def sents2t(sentences, seq_len):
    ret = torch.zeros(len(sentences), seq_len, dtype=torch.int64)
    for _id, sentence in enumerate(sentences):
        ret[_id, 0:len(sentence)] = torch.tensor(sentence, dtype=torch.int64)
    return ret


def b_sents2t(batch, seq_len):
    ret = torch.zeros(len(batch), len(batch[0]), seq_len, dtype=torch.int64)
    for bid, sentences in enumerate(batch):
        for sid, sentence in enumerate(sentences):
            ret[bid, sid, 0:len(sentence)] = torch.tensor(sentence, dtype=torch.int64)
    return ret
