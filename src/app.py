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
from src import components as c
from typing import List

'''
This is the main Programming where the Server will be started and the navigator are constructed.
'''

app = dash.Dash(__name__, use_pages=True, prevent_initial_callbacks=False, suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                )


navbar = dbc.Navbar(
            dbc.Container([
                dbc.Row(dbc.NavbarBrand("Joint Probability Trees", className="ms-2")),
                dbc.Row(dbc.NavItem(dcc.Upload(children=dbc.Button("🌱", n_clicks=0, className=""),
                                               id="upload_tree"))),
                dbc.Row([
                    dbc.Col([
                        dbc.Nav(c.gen_Nav_pages(dash.page_registry.values(), ["Empty"]), navbar=True,)
                    ])
                ], align="center")
            ]), color="dark", dark=True,
        )

def server_layout():
    '''
        Returns the Dash Strucktur of the JPT-GUI where the pages are Contained
    :return: Dash Container of the Static Page Elements
    '''
    return dbc.Container(
        [
            dbc.Row(navbar),
            dash.page_container,
            dcc.ConfirmDialog(id="tree_change_info", message="Tree was changed!"),
            dcc.Location(id="url")
        ]
    )

app.layout = server_layout


@app.callback(
    Output('tree_change_info', 'displayed'),
    Output('url', "pathname"),
    Input("upload_tree", "contents"),
)
def tree_update(upload):
    '''
        Loads a chosen jpt Tree and Refresehs to home page
        if it dosnt load nothing happens (Empty page default back to home)
    :param upload: the Paramter Dash generats from chosen a File
    :return: if the Tree was Changed and which page to load
    '''
    if upload is not None:
        try:
            content_type, content_string = upload.split(',')
            decoded = base64.b64decode(content_string)
            io_tree = jpt.JPT.from_json(json.loads(decoded))
        except Exception as e:
            print(e)
            return False, "/"
        c.in_use_tree = io_tree
        c.priors = io_tree.priors
        return True, "/empty"
    return False, "/"


if __name__ == '__main__':
    app.run_server(debug=True)

    #Dash Hover Upload beim Samen