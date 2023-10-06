import logging

from joblib import load
from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    df = datasets.load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(
        df["data"], df["target"], test_size=0.33, random_state=42
    )

    model = load("1.joblib")

    pred = model.predict(X_test)

    model_score = model.score(X_test, y_test)

    # logging.info(f"model score: {model_score:.3f}")
    print('model_score =',model_score)
    pred.tofile('pred.csv', sep = '')