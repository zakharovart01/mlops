import logging

from joblib import dump
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    df = datasets.load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(
        df["data"], df["target"], test_size=0.33, random_state=42
    )

    model = Pipeline(
        steps=[
            # ('preprocessor', preprocessor),
            ("linearregression", LinearRegression())
        ]
    )

    model.fit(X_train, y_train)

    # model_score = model.score(X_test, y_test)

    # logging.info(f"model score: {model_score:.3f}")

    dump(model, "1.joblib")
