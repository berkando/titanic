import pandas as pd
from category_encoders import SumEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_frame_imputer import DataFrameImputer
from evaluate_models import fit_predict, cross_validate_models
from mapping_transformer import MappingTransformer
from regex_extractor import RegexExtractor

TITLE_EXTRACTOR_PATTERN = r", (?P<Title>.+?)\."
NAME_TITLE_LASTNAME_PATTERN = r"(?P<Forename>.+), (?P<Title>.+)\. (?P<Lastname>.+)"

title_order1 = {
    # Dying
    'Capt': 0,
    'Rev': 0,

    # Ordinal male
    'Master': 1,
    'Mr': 1,
    'Don': 1,

    # educated  or military
    'Dr': 2,
    'Col': 2,
    'Major': 2,

    # Ordinal female
    'Miss': 4,
    'Mrs': 4,
    'Ms': 4,
    'Dona': 4,

    # Nobels
    'Jonkheer': 5,
    'Mme': 5,
    'Sir': 5,
    'Lady': 5,
    'Mlle': 5,
    'the Countess': 5,
}

title_order = {
    # Dying
    'Capt': 0,
    'Rev': 0,

    # Ordinal male
    'Master': 2,
    'Mr': 1,
    'Don': 1,

    # educated  or military
    'Dr': 2,
    'Col': 2,
    'Major': 2,

    # Ordinal female
    'Miss': 1,
    'Mrs': 1,
    'Ms': 1,
    'Dona': 1,

    # Nobels
    'Jonkheer': 2,
    'Mme': 2,
    'Sir': 2,
    'Lady': 2,
    'Mlle': 2,
    'the Countess': 2,
}


def read_data():
    drop_col = ['Ticket', 'Cabin']
    df_train = pd.read_csv('train.csv').set_index('PassengerId').drop(drop_col, axis=1)
    df_validation = pd.read_csv('test.csv').set_index('PassengerId').drop(drop_col, axis=1)

    response = 'Survived'
    features = list(set(df_train.columns) - set([response]))

    X_train = df_train[features]
    y_train = df_train[response]
    X_test = df_validation[features]

    return X_train, y_train, X_test


def get_models(pipeline):
    models = {
        'LogisticRegression': pipeline(LogisticRegression()),
        'RandomForestClassifier': pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
        'GradientBoostingClassifier': pipeline(GradientBoostingClassifier(random_state=42)),
        'GradientBoostingClassifier=1': pipeline(GradientBoostingClassifier(max_features=1, random_state=42)),
        'GradientBoostingClassifier=2': pipeline(GradientBoostingClassifier(max_features=2, random_state=42)),
        'GradientBoostingClassifier=3': pipeline(GradientBoostingClassifier(max_features=3, random_state=42)),
        'GradientBoostingClassifier=4': pipeline(GradientBoostingClassifier(max_features=4, random_state=42)),
        'GradientBoostingClassifier=5': pipeline(GradientBoostingClassifier(max_features=5, random_state=42)),
        'GradientBoostingClassifier=10': pipeline(GradientBoostingClassifier(max_features=10, random_state=42)),
        # 'XGBClassifier': xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
    }
    return models


def get_pipeline(estimator):
    categorical_features = ['Sex', 'Embarked', 'Title']
    response = 'Survived'
    df_imputer = DataFrameImputer()
    title_extractor = RegexExtractor(column='Name', regex=TITLE_EXTRACTOR_PATTERN)
    title_encoder = MappingTransformer(column='Title', mapping=title_order)
    # title_encoder = MultiColumnLabelEncoder(columns=['Title'])
    sum_encoder = SumEncoder(response, cols=categorical_features)

    pipeline = Pipeline([
        ('imputer', df_imputer),
        ('title_extractor', title_extractor),
        ('title_encoder', title_encoder),
        ('sum_encoder', sum_encoder),
        ('estimator', estimator),
    ])
    return pipeline


def export_data(X_test, y_pred, response, output_filename):
    submission = pd.DataFrame({
        X_test.index.name: X_test.index,
        response: y_pred}
    )
    submission.to_csv(output_filename, index=False)


if __name__ == '__main__':
    export = True
    X, y, X_test = read_data()

    if False:
        estimator = GradientBoostingClassifier(random_state=42)
        pipeline = get_pipeline(estimator)
        y_pred = fit_predict(X, y, X_test, pipeline)
        export_data(X_test, y_pred, 'Survived', 'submission.csv')

    else:
        cross_validate_models(X, y, X_test, get_models(get_pipeline))
