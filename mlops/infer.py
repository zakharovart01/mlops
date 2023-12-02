import logging

import gspread
import hydra
import numpy as np
import pandas as pd
from joblib import load
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg):
    logging.basicConfig(level=logging.DEBUG)

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    c = ServiceAccountCredentials.from_json_keyfile_name(
        "../token.\
    json",
        scope,
    )
    client = gspread.authorize(c)
    sheet_id = "1l3EBkmx0Mb5qHDLNiMwDKw6KlwvJyGpMP8V9GGbnCOE"
    sheet = client.open_by_key(sheet_id)
    tab_name = "data.csv"
    worksheet = sheet.worksheet(tab_name)
    data = pd.DataFrame(worksheet.get_all_values())
    logging.info("--------------------------------------")
    logging.info(data.columns)
    logging.info("--------------------------------------")
    data.columns = data.iloc[0]
    data = data.iloc[1:]
    logging.info("--------------------------------------")
    logging.info(data.shape)
    logging.info("--------------------------------------")
    logging.info(data.columns)
    logging.info("--------------------------------------")
    logging.info(cfg)
    logging.info("--------------------------------------")

    target = np.array(data["target"])
    data = np.array(data[[str(i) for i in range(10)]])

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=cfg.d.t_s, random_state=cfg.d.r_s
    )

    model = load("1.joblib")

    pred = model.predict(X_test)

    model_score = model.score(X_test, y_test)

    logging.info(f"model score: {model_score:.3f}")
    # print("model_score =", model_score)
    # если что все логи выше помогли мне отдебажить код
    # чтение данных из гугл диска взял так как я делал
    # ибо dvc api чтобы бархлил и будто только с репозитория качать может
    pred.tofile("pred.csv", sep="")


if __name__ == "__main__":
    main()
