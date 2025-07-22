from transformers.models.distilbert import configuration_distilbert

class DistilBertConfig(configuration_distilbert.DistilBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
