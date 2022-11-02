import jpt
import jpt.variables
import igraph
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from jpt.base.utils import list2interval
import dash
from dash import dcc, html, Input, Output, State, ctx, MATCH, ALLSMALLER, ALL
import math
import base64
import components as c
global model
import json
model: jpt.trees.JPT = jpt.JPT.load('test.datei')

global priors
priors = model.independent_marginals()

global result

global page
page = 0

app = dash.Dash(__name__, prevent_initial_callbacks=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

app.layout = dbc.Container(
    [
        dbc.Row(
            [
            dbc.Col(html.H1("test", className='text-center mb-4'), width=12),
            dbc.Col(dcc.Upload(children=dbc.Button("🌱", n_clicks=0, className="position-absolute top-0 end-0"),
                                   id="upload_tree"))
            ]
        ),
        dbc.Row(
            [

                dbc.Col(id="e_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_e', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid border-start border-3 border-secondary ps-3"),
                dbc.Col(id="e_input",
                        children=[dcc.Dropdown(id={'type': 'i_e', 'index': 0}, disabled=True)], width=3,
                        className="d-grid border-end border-secondary border-3 rounded-4"),
                dbc.Button("O", id=dict(type='b_e', index=0), disabled=True, n_clicks=0, size=1)
            ], className="justify-content-md-center"
        ),
    ], fluid=True
)

@app.callback(
    Output('e_variable', 'children'),
    Output('e_input', 'children'),
    Input("upload_tree", 'contents'),
    Input({'type': 'dd_e', 'index': ALL}, 'value'),
    State('e_variable', 'children'),
    State('e_input', 'children'),
)
def evid_gen(upload, dd_vals, e_var, e_in):
    e_var: list[dict] = e_var
    e_in: list[dict] = e_in
    cb = ctx.triggered_id
    if cb == "upload_tree" and upload is not None:
        global model
        global priors
        try:
            content_type, content_string = upload.split(',')
            decoded = base64.b64decode(content_string)
            io_model = jpt.JPT.from_json(json.loads(decoded))
        except Exception as e:
            print("ModelLaden hat net geklappt!")
            print(e)
            return e_var, e_in,
        e_var_n, e_in_n = c.reset_gui(io_model, "e")
        model = io_model
        priors = model.independent_marginals()

        return e_var_n, e_in_n
    elif cb.get("type") == "dd_e":
        if dd_vals[cb.get("index")] is None:
            return c.del_selector_from_div(model, e_var, e_in)
        variable = model.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
            minimum = priors[variable].cdf.intervals[0].upper
            maximum = priors[variable].cdf.intervals[-1].lower
            e_in[cb.get("index")] = c.create_range_slider(minimum, maximum, id={'type': 'i_e', 'index': cb.get("index")},
                                                        dots=False,
                                                        tooltip={"placement": "bottom", "always_visible": False})

        elif variable.symbolic:
            e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e", "index": cb.get("index")},
                                                 options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                               variable.domain.labels.values())},
                                                 value=list(variable.domain.labels.values()),
                                                 multi=True, )

        if len(e_var) - 1 == cb.get("index"):
            test1, test2 = c.add_selector_to_div(model, e_var, e_in, "dd_e", cb.get("index")+1)
            print(test1, test2)
            return test1, test2

    return c.update_free_vars_in_div(model, e_var), e_in


if __name__ == '__main__':
    app.run_server(debug=True)

