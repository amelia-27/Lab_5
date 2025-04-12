import time
import importlib

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC

import utils.dash_reusable_components as drc

# from classification-algorithm: (maybe dont need to dup)
df = pd.read_csv("./assets/Titanic-Dataset.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Pclass"] = df["Pclass"].fillna(df["Pclass"].median())

X = df[["Pclass", "Age"]]
Y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

chosen_model = LogisticRegression()
chosen_model.fit(X_train, y_train)

y_pred_tree = chosen_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_tree)

#ROC Curve
y_prob = chosen_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Titanic Survival Prediction Explorer", style={"textAlign": "center"}),
    html.H4("Amelia Ubben | CS150", style={"textAlign": "center"}),

    html.Div([
        drc.NamedSlider(
            name = "Sample Size",
            id = "slider-dataset-sample-size",
            min = 178,
            max = 890,
            step = 178,
            marks = {
                str(i): str(i)
                for i in [178, 356, 534, 712, 890]},
            value = 890,
        ),
        drc.NamedSlider(
            name="Threshold",
            id="slider-threshold",
            min=0,
            max=1,
            value=0.5,
            step=0.10,
        ),
        html.Button(
            "Reset Values",
            id="button-reset",
        ),
        ], style={"width": "30%", "padding": "10px", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            dcc.Graph(id="scatter-plot"),
        ],style={"width": "70%", "display": "inline-block"}),


        html.Div([
            dcc.Graph(id="roc-curve"),
        ], style={"width": "30%", "textAlign": "center"}),

        html.Div(id = "model-accuracy", style={"fontSize": "20px", "marginBottom": "20px"}),

        dash_table.DataTable(
            id = "confusion-table",
            columns=[
                {"name": "Actual", "id": "Actual"},
                {"name": "Predicted Negative (0)", "id": "Predicted Negative (0)"},
                {"name": "Predicated Positive (1)", "id": "Predicted Positive (1)"}
            ],
            style_table={"marginTop": "20px", "width" :"25%"}
        )

])

#CallBacks

@app.callback(
    [Output("scatter-plot", "figure"),
    Output("roc-curve", "figure"),
    Output("model-accuracy", "children"),
    Output("confusion-table", "data")],

    [Input("slider-dataset-sample-size", "value"),
     Input("slider-threshold", "value"),]
)
def update_graph(sample_size, threshold):
    #print(f"Sample Size: {sample_size}, Threshold: {threshold}")

    max_sample_size = len(X_test)
    if sample_size > max_sample_size:
        sample_size = max_sample_size

    sampled_data = X_test.copy()
    sampled_data["Actual"] = y_test.values
    sampled_data["Prob"] = y_prob
    sampled_data = sampled_data.sample(n=sample_size, random_state = 42)

    sampled_data["Predicted"] = (sampled_data["Prob"] >= threshold).astype(int)
    accuracy = accuracy_score(sampled_data["Actual"], sampled_data["Predicted"])

    sampled_data = sampled_data.dropna(subset=["Actual", "Predicted", "Age", "Pclass"])

    scatter_fig = px.scatter(
        sampled_data,
        x= "Age",
        y = "Pclass",
        color = "Actual",
        symbol = "Predicted",
        labels={"Actual": "Actual Survival", "Pclass": "Passenger Class"},
        title = "Survival by Age and Passenger Class"
    )
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name = f"ROC Curve (AUC = {roc_auc:.2f})"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 line=dict(dash='dash', color='red'),
                                 name = 'Random Chance'))
    roc_fig.update_layout(title='ROC Curve',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate')

    confuse_matrix = confusion_matrix(y_test, sampled_data["Predicted"])
    tn, fp, fn, tp = confuse_matrix.ravel()
    table_data = [
        {"Actual": "Negative (0)", "Predicted Negative (0)": tn, "Predicted Positive (1)": fp},
        {"Actual": "Positive (1)", "Predicted Negative (0)": fn, "Predicted Positive (1)": tp},

    ]

    return scatter_fig, roc_fig, f"Model Accuracy: {accuracy * 100: .2f}%", table_data


@app.callback(
    [Output("slider-threshold", "value"),
     Output("slider-dataset-sample-size", "value")],
    [Input("button-reset", "n_clicks")],
    [State("slider-threshold", "value"),
     State("slider-dataset-sample-size", "value")],
)
def reset_sliders(n_clicks, current_threshold, current_sample_size):
    if n_clicks:
        return 0.5, 888

    return current_threshold, current_sample_size


if __name__ == "__main__":
    app.run(debug=True)
