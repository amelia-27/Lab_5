#This file is for Part 3 of the Lab
#Prints ROC curve and model accuracy
#After investigation, Logistic Regression will work the best for my data.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html

df = pd.read_csv("../assets/Titanic-Dataset.csv")
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Pclass"] = df["Pclass"].fillna(df["Pclass"].median())

X = df[["Pclass", "Age"]]
Y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

chosen_model = LogisticRegression()
chosen_model.fit(X_train, y_train)

# Make predictions and compute accuracy
y_pred_tree = chosen_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_tree)
print("Model Accuracy:", accuracy)

#ROC Curve
y_prob = chosen_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

# Create a Plotly ROC curve figure
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'ROC Curve (AUC = {roc_auc:.2f})'))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             line=dict(dash='dash', color='red'),
                             name='Random Chance'))
roc_fig.update_layout(title='ROC Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')

# Create a scatter plot for Age vs. Gender (color-coded by Actual survival)
data_plot = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred_tree,
    "Pclass": X_test["Pclass"],
    "Age": X_test["Age"]
})
scatter_fig = px.scatter(data_plot, x="Age", y="Pclass", color="Actual",
                         title="Age vs. Ticket Class by Survival",
                         labels={"Actual": "Survival (0 = No, 1 = Yes)"})

# Build the Dash app layout with both graphs
app = Dash(__name__)
app.layout = html.Div([
    html.Div(f"Model Accuracy: {accuracy * 100:.2f}%"),
    dcc.Graph(figure=scatter_fig),
    dcc.Graph(figure=roc_fig)
])

if __name__ == "__main__":
    app.run(debug=True)