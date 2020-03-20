from transformers import (AlbertConfig, AlbertModel, AlbertTokenizer,
                          BertConfig, BertModel, BertTokenizer,
                          DistilBertConfig, DistilBertModel,
                          DistilBertTokenizer, FlaubertConfig, FlaubertModel,
                          FlaubertTokenizer, RobertaConfig, RobertaModel,
                          RobertaTokenizer, XLMConfig, XLMModel,
                          XLMRobertaConfig, XLMRobertaModel,
                          XLMRobertaTokenizer, XLMTokenizer, XLNetConfig,
                          XLNetModel, XLNetTokenizer)

transformer_classes = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "xlm": (XLMConfig, XLMModel, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertModel, FlaubertTokenizer),
}


def TransformerConfig(model_name):
    for name, classes in transformer_classes.items():
        if model_name.startswith(name):
            return classes[0]


def TransformerModel(model_name):
    for name, classes in transformer_classes.items():
        if model_name.startswith(name):
            return classes[1]


def TransformerTokenizer(model_name):
    for name, classes in transformer_classes.items():
        if model_name.startswith(name):
            return classes[2]
