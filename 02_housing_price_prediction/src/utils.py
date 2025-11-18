import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def plot_regression_results(y_true, y_pred, title='Regression Results'):
    """
    Plots a scatter plot of true vs. predicted values with a clear legend.
    """
    
    # Calculate R^2 score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    fig = px.scatter(x=y_true, y=y_pred, 
                     labels={'x': 'True Prices ($)', 'y': 'Predicted Prices ($)'},
                     title=f'{title} (R² Score: {r2:.4f})',
                     trendline='ols', # Adds a "best fit" line
                     opacity=0.7,
                     )
    
    try:
        fig.data[1].name = 'OLS Trendline (Best Fit)'
        fig.data[1].showlegend = True
    except (IndexError, AttributeError):
        pass # Kontynuuj, jeśli z jakiegoś powodu linia trendu nie powstała


    fig.add_shape(type='line',
                  x0=y_true.min(), y0=y_true.min(),
                  x1=y_true.max(), y1=y_true.max(),
                  line=dict(color='Red', dash='dash'))

  
    fig.add_trace(go.Scatter(
        x=[None], y=[None], # Brak danych
        mode='lines',
        line=dict(color='Red', dash='dash'),
        name='Perfect Prediction (y=x)' # Etykieta do legendy
    ))
    

    fig.update_layout(width=800, height=600, showlegend=True) # Wymuś pokazanie legendy
    fig.show()

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots the most important features for a tree-based model.
    """
    if not hasattr(model, 'feature_importances_'):
        print("Cannot plot feature importance for this model type (no 'feature_importances_').")
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