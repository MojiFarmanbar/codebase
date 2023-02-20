import logging
from pathlib import Path
import typer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib
from bank.data import load_data, import_config
from bank.feature import categorical_feature_mapper, numerical_feature_mapper


app = typer.Typer()


@app.callback()
def main() -> None:
    """Determine customer subscription outcomes."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)-15s] %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def train(input_path: Path, model_path: Path) -> None:
    """Trains a model on the given dataset."""

    logger = logging.getLogger(__name__)

    logger.info("Loading input dataset from %s", input_path)
    train_dataset = load_data(input_path)
    logger.info("Found %i rows", len(train_dataset))

    # TODO: Fill in your solution.
    logger.info('Mapping the features')
    categorical_feature = categorical_feature_mapper(train_dataset)
    numerical_feature = numerical_feature_mapper(train_dataset)
    df_mapped = pd.concat([numerical_feature, categorical_feature], axis=1)

    # - Separate X from y
    X = pd.DataFrame(df_mapped.drop(['y'], axis=1))
    y = df_mapped['y']

    # - Fit a model
    logger.info('Fitting the model')
    feature = ColumnTransformer([
    ("numeric", StandardScaler(), ['age', 'cons.price.idx',
     'cons.conf.idx', 'euribor3m', 'previous', 'pdays',
     'log_duration','log_campaign','day_of_week1_sin',
     'month1_cos']),
    ("categorical", OneHotEncoder(sparse=False), ['marital','job', 'education', 'poutcome'])]
        , remainder='drop'
    )

    model_config = import_config()
    pipe = Pipeline([
        ("feature", feature), 
        ("classifier", RandomForestClassifier(n_estimators=model_config['model_parameters']['n_estimators'], 
        max_depth=model_config['model_parameters']['max_depth'], 
        max_features=model_config['model_parameters']['max_features']))
    ])

    # #Remove the target variable from X and make Y = target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = pipe.fit(X_train, y_train)
    final_score = metrics.accuracy_score(y_test, clf.predict(X_test))

    # - Log the final score
    logger.info(f"This is the final score {final_score}")

    # - Save model
    joblib.dump(clf, f'{model_path}' )
    logger.info(f"Wrote model to {model_path}")


@app.command()
def predict(input_path: Path, model_path: Path, output_path: Path):
    "Make predictions for the given dataset"

    logger = logging.getLogger(__name__)
    logger.info("Loading input dataset from %s", input_path)
    test_dataset = load_data(input_path)
    logger.info("Found %i rows", len(test_dataset))

    logger.info(f"Loading model from {model_path}")
    outcome_model = joblib.load(f'{model_path}')

    logger.info("Making predictions:")
    categorical_feature = categorical_feature_mapper(test_dataset)
    numerical_feature = numerical_feature_mapper(test_dataset)
    df_mapped = pd.concat([numerical_feature, categorical_feature], axis=1)
    y_pred = outcome_model.predict_proba(pd.DataFrame(df_mapped.drop(['y'], axis=1)))

    logger.info("Saving predictions")
    classes = outcome_model.classes_.tolist()
    proba_df = pd.DataFrame(y_pred, columns=classes)
    rendered = pd.concat([test_dataset,proba_df], axis=1)
    rendered.to_csv(f'{output_path}', index=False)