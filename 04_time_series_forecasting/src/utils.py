import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def plot_forecast_vs_actual(y_true, y_pred, title='Forecast vs. Actuals'):
    """
    Plots the actual values vs. the forecasted values over time.
    
    Args:
    y_true (pd.Series): Series with a DateTimeIndex.
    y_pred (pd.Series): Series with a DateTimeIndex.
    title (str): Title of the plot.
    """
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Actual': y_true,
        'Forecast': y_pred
    })
    
    fig = px.line(plot_df, title=title)
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        legend_title="Series"
    )
    fig.show()

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots the most important features for a tree-based model (like LightGBM).
    """
    if not hasattr(model, 'feature_importances_'):
        print("Cannot plot feature importance for this model type.")
        return
        
    # Get importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    
    # Sort and get top N
    top_df = importance_df.nlargest(top_n, 'importance').sort_values(by='importance', ascending=True)
    
    fig = px.bar(top_df, 
                 x='importance', 
                 y='feature', 
                 orientation='h',
                 title=f'Top {top_n} Most Important Features')
    
    fig.update_layout(yaxis_title="Feature", xaxis_title="Importance")
    fig.show()