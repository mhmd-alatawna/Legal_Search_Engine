class Schema:
    def __init__(self, fields, primary_keys, main_field, vector_size = None):
        self.fields = fields
        self.primary_keys = primary_keys
        self.main_field = main_field
        self.vector_size = vector_size

    def __eq__(self, other):
        if tuple(self.fields) != tuple(other.fields) :
            return False
        if tuple(self.primary_keys) != tuple(other.primary_keys):
            return False
        if self.main_field != other.main_field:
            return False
        if self.vector_size != other.vector_size:
            return False
        return True