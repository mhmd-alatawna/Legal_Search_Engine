import copy

from src.DatabaseLayer.Processors.BaseProcessor import BaseProcessor


class CopyProcessor(BaseProcessor):
    schema = 0
    def __init__(self):
        super().__init__()
        self.schema = copy.deepcopy(CopyProcessor.schema)

    def extract(self, data_tuple):
        return [data_tuple]