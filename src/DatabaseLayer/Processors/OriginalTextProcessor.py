from src.DatabaseLayer.Processors.BaseProcessor import BaseProcessor
from src.DatabaseLayer.Processors.Schema import Schema


class OriginalTextProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.fields = (("case_id","TEXT"), ("original_text","TEXT"), ("citations","TEXT"), ("test","TEXT"))
        self.primary_keys = ("case_id",)
        self.schema = Schema(self.fields, self.primary_keys, "original_text")

    def extract(self, data_tuple):
        return [(data_tuple[0], data_tuple[1], data_tuple[2], data_tuple[3])]