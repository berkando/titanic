from sklearn.base import TransformerMixin


class MappingTransformer(TransformerMixin):
    def __init__(self, column=None, mapping=None):
        super(MappingTransformer, self).__init__()
        self.mapping = mapping
        self.column = column

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, X, y=None):
        X_out = X.copy(deep=True)
        X_out[self.column] = X_out[self.column].map(self.mapping)
        return X_out