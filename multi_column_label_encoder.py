from sklearn.preprocessing import LabelEncoder


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode
        self.label_enc = {}

    def fit(self, X, y=None):
        self.label_enc = {col: LabelEncoder().fit(X[col]) for col in self.columns}
        return self

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        for col in self.columns:
            output[col] = self.label_enc[col].transform(output[col])
        return output