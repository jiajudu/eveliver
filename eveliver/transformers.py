from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer, XLMConfig, XLMForSequenceClassification, XLMTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer, AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer, FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer
transformer_classes = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}