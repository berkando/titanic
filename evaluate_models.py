import pandas as pd
from sklearn.model_selection import cross_val_score


def eval_models(models, X, y, cv=2):
    result = [
        (_cross_val_score(X, cv, model, name, y), name, model,)
        for name, model in models.items()
    ]
    return pd.DataFrame(result,
                        columns=['score', 'model_name', 'model'],
                        ).sort_values(by='score', ascending=False)


def _cross_val_score(X, cv, model, name, y):
    print('~~~ {} '.format(name))
    score = cross_val_score(model, X, y, cv=cv, n_jobs=1).mean()
    print(score)
    return score


def fit_predict(X, y, X_test, pipeline):
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X_test)
    return y_pred


def cross_validate_models(X, y, X_test, models):
    e_models = eval_models(models, X, y)
    print('-' * 40)
    print(e_models[['model_name', 'score']])

    print('-' * 40)
    best_model = e_models.model[e_models.index[0]]

    best_model.fit(X, y)
    y_pred = best_model.predict(X_test)
    return y_pred
