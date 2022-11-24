import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html, no_update
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import base64
import os
import glob

from sqlalchemy import engine_from_config

# Create dash app
app = dash.Dash(__name__)

# read data
os.chdir("..\..")
os.chdir("output\ovary20221121")
res = pd.read_csv(r"result.csv")
trials = res.shape[0]
res['idx'] = [str(i) if len(str(i)) == 2 else "0" + str(i) for i in res.idx]
#res = res.sort_values(by = 'idx', key = lambda col: col.astype(str))
res = res.sort_values(by = 'idx')

# find images
images = list(glob.glob("*val_curve.png"))
full_path = [os.path.join(os.getcwd(), img) for img in images] 
encoded_imgs = [base64.b64encode(open(img, 'rb').read()).decode('ascii') for img in full_path]
puppy = os.path.join(os.getcwd(), "puppy.png")
enc_puppy = base64.b64encode(open(puppy, 'rb').read()).decode('ascii')

#encoded_imgs = encoded_imgs + [enc_puppy]*trials
# Generate dataframe
df = pd.DataFrame(
   dict(
      error=[*res['outAE'], *res['outNMF']],
      cosine=[*res['AE_perm'],*res['NMF_perm']],
      method = ['AE']*trials + ['NMF']*trials,
      )
)

# Create scatter plot with x and y coordinates

fig = px.scatter(df, x="error", y="cosine", color = "method")

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
    print(pt)

    #ved ikke hvor curve number kommer fra, men det har noget med "method at gøre"
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    curve_num = pt["curveNumber"]
    #img_url = df['img'][num]
    img_url = encoded_imgs[num] if curve_num == 0 else enc_puppy # vis puppy for NMF (curveNum = 1)
    
    children = [
        html.Div([
            html.Img(
                src='data:image/png;base64,{}'.format(img_url),
                style={'width': '300px', 'white-space': 'none','marginBottom': '0px', 'marginTop': '0px'},
            ),
            html.P("point_nr " + str(num), style={'font-weight': 'bold'})
        ])
    ]

    return True, bbox, children
   
if __name__ == '__main__':
   app.run_server(debug=True)
