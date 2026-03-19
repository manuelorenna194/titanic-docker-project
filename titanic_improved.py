import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["LogFare"] = np.log1p(df["Fare"])
    return df


def build_pipeline() -> Pipeline:
    numeric_features = ["Age", "Fare", "LogFare", "SibSp", "Parch", "FamilySize", "IsAlone"]
    categorical_features = ["Sex", "Pclass", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf


def main() -> None:
    data = load_data("data/train_and_test2.csv")

    # Adapt to this dataset's column names
    data = data.rename(
        columns={
            "Passengerid": "PassengerId",
            "sibsp": "SibSp",
            "Parch": "Parch",
            "2urvived": "Survived",
        }
    )

    # Minimal subset of useful columns (others in this file are one-hot zeros)
    features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
    target = "Survived"

    data = data[features + [target]]

    X = data[features]
    y = data[target]

    X = add_features(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()

    param_distributions = {
        "model__max_depth": [None, 4, 6, 8, 12],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring=make_scorer(f1_score, pos_label=1),
        n_jobs=-1,
        random_state=42,
    )

    search.fit(x_train, y_train)

    print("Best params:", search.best_params_)
    print("Best CV score:", search.best_score_)

    y_pred = search.predict(x_test)
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Example: predict for a likely-to-survive passenger
    example = pd.DataFrame(
        [[1, 1, 25, 80.0, 0, 0, 0]],
        columns=["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"],
    )
    example = add_features(example)
    example_pred = search.predict(example)
    print(
        "\nExample passenger prediction (should be more likely to survive):",
        "Survived" if example_pred[0] == 1 else "Did not survive",
    )


if __name__ == "__main__":
    main()

