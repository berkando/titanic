import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    fill = None

    def fit(self, X, y=None, **fit_params):
        def fill_values(x):
            if x.dtype == np.dtype('O'):
                return x.value_counts().index[0]
            elif x.dtype == np.dtype('float') or x.dtype == np.dtype('int'):
                return x.median()
            raise Exception('Unknown datatype')

        data = {c: fill_values(X[c]) for c in X}
        self.fill = pd.Series(data, index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)