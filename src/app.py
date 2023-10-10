import base64
import os.path

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
import sys, getopt

'''
This is the main Programming where the Server will be started and the navigator are constructed.
'''

pre_tree = ""
app_tags = dict(debug=True, dev_tools_hot_reload=False)
if len(sys.argv) > 1:
    opts, args = getopt.getopt(sys.argv[1:], "t:h:p:", ["tree=","host=", "port=", "help"])
    for opt, arg in opts:
        if opt in ("-t", "--tree"):
            if not os.path.isfile(arg):
                raise ValueError(f"file {arg} dose not exist.")
            pre_tree = arg
        elif opt in ("-h", "--host"):
            app_tags.update(dict(host=str(arg)))
        elif opt in ("-p", "--port"):
            app_tags.update(dict(port=int(arg)))
        elif opt == "--help":
            print("-t, --tree you can preload a tree with its path from the app.py directory \n -h, --host you can change the IP of the GUI \n -p --port you can change the port of the GUI \n Default Address is (http://127.0.0.1:8050/)")
            exit(0)

app = dash.Dash(__name__, use_pages=True, prevent_initial_callbacks=False, suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],
                )


navbar = dbc.Navbar(
            dbc.Container([
                dbc.Row(dbc.Col(html.Img(src="./assets/Logo_JPT_White_s.png", height="50px"), className="ps-4")),
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
            content_decided_string = decoded.decode("utf-8")
            io_tree = jpt.JPT.from_json(json.loads(decoded))
        except Exception as e:
            print(e)
            return False, "/"
        c.in_use_tree = io_tree
        c.priors = io_tree.priors
        return True, "/empty"
    return False, "/"


if __name__ == '__main__':
    if pre_tree != "":
        try:
            tree = open(pre_tree, "rb")
            tree_data = tree.read()
            io_tree = jpt.JPT.from_json(json.loads(tree_data))
            c.in_use_tree = io_tree
            c.priors = io_tree.priors
            tree.close()
        except Exception:
            print("File could not be read")
            exit(1)

    app.run(**app_tags)



