import logging

import gspread
import hydra
import mlflow
import mlflow.onnx
import numpy as np
import onnxruntime as rt
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split


def make_metrics(y_true, y_pred):
    metrics = {}
    rmse = np.mean((y_true - y_pred) ** 2) ** 0.5
    mape = np.mean(np.abs(y_true - y_pred) / (y_true + 1e-6))
    max_error = np.max(np.abs(y_true - y_pred))

    metrics = {"RMSE": rmse, "MAPE": mape, "max error": max_error}
    return metrics


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg):
    logging.basicConfig(level=logging.DEBUG)

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    c = ServiceAccountCredentials.from_json_keyfile_name(
        "../token.json",
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

    target = np.array(data["target"]).astype(np.float32)
    data = np.array(data[[str(i) for i in range(10)]]).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=cfg.d.t_s, random_state=cfg.d.r_s
    )

    # model = load("1.joblib")
    sess = rt.InferenceSession("model.onnx")

    # pred = model.predict(X_test)
    pred = np.array(sess.run(None, {"input": X_test}))

    metrics = make_metrics(y_test, pred)
    # signature = infer_signature(X_test, y_test)

    # logging.info(f"model score: {model_score:.3f}")
    # print("model_score =", model_score)
    # если что все логи выше помогли мне отдебажить код
    # чтение данных из гугл диска взял так как я делал
    # ибо dvc api чтобы бархлил и будто только с репозитория качать может
    pred.tofile("pred.csv", sep="")
    mlflow.log_params(cfg)
    mlflow.log_metrics(metrics)
    logging.info("--------------------------------------")
    logging.info(metrics)
    logging.info("--------------------------------------")
    # mlflow.sklearn.log_model(model,
    # signature=signature, artifact_path="1.joblib")


if __name__ == "__main__":
    main()
