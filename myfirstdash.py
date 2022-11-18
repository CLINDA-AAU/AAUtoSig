import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html, no_update
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import base64
import os
import glob

# Create dash app
app = dash.Dash(__name__)

# read data
res = pd.read_csv(r"result.csv")
trials = res.shape[0]
# find images
images = list(glob.glob("*val_curve.png"))
full_path = [os.path.join(os.getcwd(), img) for img in images] 
encoded_imgs = [base64.b64encode(open(img, 'rb').read()).decode('ascii') for img in full_path]

# Generate dataframe
df = pd.DataFrame(
   dict(
      x=[*res['outAE'], *res['outNMF']],
      y=[*res['AE_perm'],*res['NMF_perm']],
      method = ['AE']*trials + ['NMF']*trials
      )
)

# Create scatter plot with x and y coordinates


fig = px.scatter(df, x="x", y="y", color = "method")

# Update layout and update traces
#fig.update_layout(clickmode='event+select')
#fig.update_traces(marker_size=20)
fig.update_traces(hoverinfo="none", hovertemplate=None)

# Create app layout to show dash graph
app.layout = html.Div(
   [
      dcc.Graph(
         id="graph-basic-2",
         figure=fig,
      ),
      dcc.Tooltip(
         id="graph-tooltip"
      )
   ]
)

# html callback function to hover the data on specific coordinates
@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("graph-basic-2", "hoverData"),
)

def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    img_url = encoded_imgs[num]
    
    children = [
        html.Div([
            html.Img(
                src='data:image/png;base64,{}'.format(img_url),
                style={'width': '300px', 'white-space': 'none','marginBottom': '0px', 'marginTop': '0px'},
            ),
        ])
    ]

    return True, bbox, children
   
if __name__ == '__main__':
   app.run_server(debug=True)
