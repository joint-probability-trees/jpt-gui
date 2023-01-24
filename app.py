import base64

import jpt
import jpt.distributions.univariate
import igraph
import numpy as np
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from jpt.base.utils import list2interval
import dash
from dash import dcc, html, Input, Output, State, ctx, MATCH, ALLSMALLER, ALL
import math
import json
import components as c
from typing import List



app = dash.Dash(__name__, use_pages=True, prevent_initial_callbacks=False, suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                )


navbar = dbc.Navbar(
            dbc.Container([
                dbc.Row(dbc.NavbarBrand("JPT", className="ms-2")),
                dbc.Row(dbc.NavItem(dcc.Upload(children=dbc.Button("🌱", n_clicks=0, className=""),
                                               id="upload_tree"))),
                dbc.Row([
                    dbc.Col([
                        dbc.Nav([
                           dbc.NavItem(dbc.NavLink(f"{page['name']}", href=page["relative_path"]))
                            for page in dash.page_registry.values()],
                            navbar=True, )
                    ])
                ], align="center")
            ]), color="dark", dark=True,
        )

def serve_layout():
    return dbc.Container(
        [
            dbc.Row(navbar),
            dash.page_container,
            dcc.ConfirmDialog(id="tree_change_info", message="Tree was changed!"),
            dcc.Location(id="url")
        ]
    )

app.layout = serve_layout


@app.callback(
    Output('tree_change_info', 'displayed'),
    Output('url', "pathname"),
    Input("upload_tree", "contents"),
)
def tree_update(upload):
    if upload is not None:
        try:
            content_type, content_string = upload.split(',')
            decoded = base64.b64decode(content_string)
            io_tree = jpt.JPT.from_json(json.loads(decoded))
        except Exception as e:
            print(e)
            return False, "/"
        c.in_use_tree = io_tree
        c.priors = io_tree.independent_marginals()
        return True, "/empty"
    return False, "/"


if __name__ == '__main__':
    app.run_server(debug=True)