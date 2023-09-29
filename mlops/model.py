from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from joblib import load

data = datasets.load_diabetes()

model = Pipeline(steps=[
#        ('preprocessor', preprocessor),
        ('linearregression', LinearRegression())
        ])

model.fit(data['data'], data['target'])

