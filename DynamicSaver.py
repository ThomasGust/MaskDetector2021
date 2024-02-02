class DynamicSaver(object):

    def __init__(self, path, iota):
        self.path = path
        self.iota = iota

    def save(self, model):
        model.save(filepath=self.path, overwrite=True)