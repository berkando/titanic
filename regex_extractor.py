import pandas as pd
from sklearn.base import TransformerMixin


class RegexExtractor(TransformerMixin):
    def __init__(self, column=None, regex=None):
        super(RegexExtractor, self).__init__()
        self.regex = regex
        self.column = column

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        X_out = X.copy(deep=True)
        data = X_out[self.column].str.extract(self.regex, expand=True)
        X_out = X_out.drop(self.column, axis=1)
        df = pd.concat([X_out, data], axis=1)
        return df