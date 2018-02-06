import pandas as pd
from sklearn.base import TransformerMixin


class OrdinalEncoder(TransformerMixin):

    def __init__(self, response, include=None, exclude=None):
        super(OrdinalEncoder, self).__init__()
        self.response = response
        self.include = set(include) if include else None
        self.exclude = set(exclude) if exclude else set()
        self._coding_map = None

    def fit(self, X, y, **fit_params):
        def sum_coding(X, y):
            df = pd.concat([X, y], axis=1)
            df_grouped_sum = df.groupby(X.name)[self.response].sum()
            return df_grouped_sum / df_grouped_sum.max()

        include = (self.include or set(X.columns)) - self.exclude
        response = pd.DataFrame(y, columns=[self.response])
        self._coding_map = {c: sum_coding(X[c], response) for c in X if c in include}

        return self

    def transform(self, X, y=None):
        include = (self.include or set(X.columns)) - self.exclude
        ret_val = pd.concat([
            X[c] if c not in include else X[c].map(self._coding_map[c])
            for c in X], axis=1)
        return ret_val