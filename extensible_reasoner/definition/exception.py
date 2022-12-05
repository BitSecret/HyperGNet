class RuntimeException(Exception):

    def __init__(self, label, note):
        self.label = label
        self.note = note

    def __str__(self):
        return "<{}> {}".format(self.label, self.note)

