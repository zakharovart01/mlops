from joblib import load
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

data = datasets.load_diabetes()

model = Pipeline(
    steps=[
        #        ('preprocessor', preprocessor),
        ("linearregression", LinearRegression())
    ]
)

model.fit(data["data"], data["target"])
