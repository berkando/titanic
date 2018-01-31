from math import log, ceil

import numpy as np
import pandas as pd
from category_encoders import SumEncoder, BinaryEncoder
from sklearn.base import TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


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


class SumEncoder1(TransformerMixin):

    def __init__(self, response, include=None, exclude=None):
        super(SumEncoder, self).__init__()
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


class BinaryEncoder1(TransformerMixin):

    def __init__(self, include=None, exclude=None):
        super(BinaryEncoder, self).__init__()
        self.include = set(include) if include else None
        self.exclude = set(exclude) if exclude else set()
        self._coding_map = None

    def fit(self, X, y=None, **fit_params):
        def binary_coding(X):
            return {value: cnt for cnt, value in enumerate(sorted(X.unique()))}

        include = (self.include or set(X.columns)) - self.exclude
        self._coding_map = {c: binary_coding(X[c]) for c in X if c in include}
        return self

    def transform(self, X, y=None):
        data = {}
        include = (self.include or set(X.columns)) - self.exclude
        for c in X:
            if c in include:
                num_bins = int(ceil(log(len(self._coding_map[c]), 2)))
                for bin in range(num_bins):
                    code = 2 ** bin
                    name = '{}_{}'.format(c, code)
                    data[name] = (X[c].map(self._coding_map[c]) & code).map(np.sign).fillna(0.0)

            else:
                data[c] = X[c]
        return pd.DataFrame(data)


class TitanicNameExtractor(TransformerMixin):

    def __init__(self):
        super(SumEncoder, self).__init__()

    def fit(self, X, y, **fit_params):
        features = {'Surname': X.Name.str.split(',').apply(lambda x: x[0].strip().upper()),
                    'Forename': X.Name.str.split(',').apply(lambda x: x[1].split('.')[1].strip().upper()),
                    'Title': X.Name.str.split(',').apply(lambda x: x[1].split('.')[0].strip().upper()),
                    }

        return self

    def transform(self, X, y=None):
        include = (self.include or set(X.columns)) - self.exclude
        ret_val = pd.concat([
            X[c] if c not in include else X[c].map(self._coding_map[c])
            for c in X], axis=1)
        return ret_val


def read_data():
    drop_col = ['Ticket', 'Cabin', 'Name']
    df_train = pd.read_csv('train.csv').set_index('PassengerId').drop(drop_col, axis=1)
    df_validation = pd.read_csv('test.csv').set_index('PassengerId').drop(drop_col, axis=1)

    dfi = DataFrameImputer()
    df_train = dfi.fit_transform(df_train)
    df_validation = dfi.transform(df_validation)

    return df_train, df_validation


def get_models():
    models = {
        # 'LogisticRegression': LogisticRegression(),
        # 'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        # 'GradientBoostingClassifier=1': GradientBoostingClassifier(max_features=1, random_state=42),
        # 'GradientBoostingClassifier=2': GradientBoostingClassifier(max_features=2, random_state=42),
        # 'GradientBoostingClassifier=3': GradientBoostingClassifier(max_features=3, random_state=42),
        # 'GradientBoostingClassifier=4': GradientBoostingClassifier(max_features=4, random_state=42),
        # 'GradientBoostingClassifier=5': GradientBoostingClassifier(max_features=5, random_state=42),
        # 'XGBClassifier': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    }
    return models


def eval_models(models, df, df_validation):
    response = 'Survived'
    features = list(set(df.columns) - set([response]))
    X = df[features]
    y = df[response]

    result = []
    for name, model in models.items():
        print('~~~ {} '.format(name))

        score = cross_val_score(model, X, y, cv=5).mean()
        print(score)
        result.append((score, name, model))
    best_score, best_model_name, best_model = max(result)
    print('Best model:')
    print(best_model_name)
    print(best_score)
    print(best_model)

    best_model.fit(X, y)
    y_pred = best_model.predict(df_validation[features])
    return y_pred, pd.DataFrame([_[:2] for _ in result],
                                columns=['best_score', 'best_model_name'],
                                ).sort_values(by='best_score', ascending=False)


def main():
    df, df_validation = read_data()

    categorical_features = ['Sex', 'Embarked', 'Ticket']

    encoder = {
        'SumEncoder': SumEncoder('Survived', cols=categorical_features),
        'BinaryEncoder': BinaryEncoder(cols=categorical_features),
    }

    pipelines = {
        '{}_{}'.format(enc_name, model_name): Pipeline([
            (enc_name, enc),
            (model_name, model)
        ])
        for model_name, model in get_models().items()
        for enc_name, enc in encoder.items()
    }
    y_pred, e_models = eval_models(pipelines, df, df_validation)
    submission = pd.DataFrame({'PassengerId': df_validation.index, # ['PassengerId'],
                               'Survived': y_pred})
    # print(submission.head())
    print(e_models)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
