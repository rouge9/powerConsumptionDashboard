import xgboost as xgb
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


external_stylesheets = [dbc.themes.CYBORG]


start = str((dt.datetime.now().date()))
end = str((dt.datetime.now().date() + dt.timedelta(days=1)))

# create a function that update the date once a day


def update_date():
    global start, end
    start = str((dt.datetime.now().date()))
    end = str((dt.datetime.now().date() + dt.timedelta(days=1)))


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df["date"] = df.index
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["second"] = df["date"].dt.second
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day

    X = df[
        [
            "hour",
            "minute",
            "second",
            "dayofweek",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
        ]
    ]

    if label:
        y = df[label]
        return X, y
    return X


def prepare_data(start_d=start, end_d=end):
    model2 = xgb.XGBRegressor()
    model2.load_model("final_model\model_sklearn.txt")
    forecast_index = pd.date_range(start=start_d, end=end_d, freq="5min")
    test = pd.DataFrame(index=forecast_index)
    test["Consumption(watt-hour)"] = 0

    X_test, y_test = create_features(test, label="Consumption(watt-hour)")
    y_pred = model2.predict(X_test)
    test["prediction"] = model2.predict(X_test)
    test[test["prediction"] < 0] = 0
    return px.line(
        test,
        x=test.index,
        y="prediction",
        title="Prediction",
        labels={"x": "Time", "y": "Consumption(watt-hour)"},
        template="plotly_dark",
    )


app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.H1("DanEnergy Power Consumption Prediction"),
        dcc.Graph(id="graph"),
        dcc.DatePickerRange(id="input-1"),
        html.Div(id="number-output"),
    ]
)


@app.callback(
    Output("graph", "figure"),
    [Input("input-1", "start_date"), Input("input-1", "end_date")],
)
def update_graph(start_date, end_date):
    if start_date is None:
        start_date = start
    if end_date is None:
        end_date = end
    return prepare_data(start_date, end_date)


if __name__ == "__main__":
    app.run_server(debug=True)
