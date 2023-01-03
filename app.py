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

global model
model: jpt.trees.JPT = c.get_tree()


app = dash.Dash(__name__, use_pages=True, prevent_initial_callbacks=True, suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}],

                )


navbar = dbc.Navbar(
            dbc.Container([
                dbc.Row(dbc.NavbarBrand("JPT", className="ms-2")),
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

app.layout = dbc.Container(
    [
        dbc.Row(navbar),
        dash.page_container

    ]
)



if __name__ == '__main__':
    app.run_server(debug=True)