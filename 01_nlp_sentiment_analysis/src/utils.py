import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np

def plot_confusion_matrix(cm, labels=['negative', 'positive']):
    """
    Plots an interactive confusion matrix using Plotly.
    """
    cm_reversed = cm[::-1]
    labels_reversed = labels[::-1]
    
    cm_df = pd.DataFrame(cm_reversed, columns=labels, index=labels_reversed)

    fig = ff.create_annotated_heatmap(
        z=cm_df.values, x=list(cm_df.columns), y=list(cm_df.index),
        colorscale='ice', showscale=True, reversescale=True
    )
    fig.update_layout(width=500, height=500, title='Confusion Matrix', font_size=16)
    fig.show()

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots the most important features (words) for a linear model.
    """
    if not hasattr(model, 'coef_'):
        print("Cannot plot feature importance for this model type (no 'coef_' attribute).")
        return
        
    # Get coefficients
    coef = model.coef_[0]
    
    # Create DataFrame of features and their coefficients
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
    
    # Get top N positive and top N negative coefficients
    top_positive = coef_df.nlargest(top_n, 'coefficient')
    top_negative = coef_df.nsmallest(top_n, 'coefficient').sort_values(by='coefficient', ascending=True)
    
    # Concatenate and create the plot
    plot_df = pd.concat([top_negative, top_positive]).sort_values(by='coefficient')
    plot_df['color'] = (plot_df['coefficient'] > 0).map({True: 'Positive', False: 'Negative'})
    
    fig = px.bar(plot_df, 
                 x='coefficient', 
                 y='feature', 
                 color='color',
                 orientation='h',
                 title=f'Top {top_n} Most Important Features (Words)',
                 color_discrete_map={'Positive': 'green', 'Negative': 'red'})
    
    fig.update_layout(yaxis_title="Feature (Word)", xaxis_title="Coefficient (Impact)")
    fig.show()