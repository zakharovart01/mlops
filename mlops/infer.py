import logging
import dvc.api
import pandas as pd

from joblib import load
from sklearn import datasets
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    df = datasets.load_diabetes()
    #https://docs.google.com/spreadsheets/d/1l3EBkmx0Mb5qHDLNiMwDKw6KlwvJyGpMP8V9GGbnCOE/edit#gid=397432644
    with dvc.api.open_url('https://docs.google.com/spreadsheets/d/1l3EBkmx0Mb5qHDLNiMwDKw6KlwvJyGpMP8V9GGbnCOE/edit#gid=397432644') as f:
        data = pd.read_csv(f)
        target = data['target']
        data = data.drop(columns=['target'], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.33, random_state=42
    )

    model = load("1.joblib")

    pred = model.predict(X_test)

    model_score = model.score(X_test, y_test)

    # logging.info(f"model score: {model_score:.3f}")
    print("model_score =", model_score)
    pred.tofile("pred.csv", sep="")
