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

global model
model = jpt.trees.JPT.load("test.datei")
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
                dbc.Col(html.H1("Posterior", className='text-center mb-4'), width=12),
                dbc.Col(dcc.Upload(children=dbc.Button("🌱", n_clicks=0, className="position-absolute top-0 end-0"),
                                   id="upload_tree"))
            ]
        ),
        dbc.Row(
            [
                dbc.Col([
                    html.Div("P ", className="ps-3",
                             style={'fontSize': 30, 'padding-top': 0}),
                ], id="text_l", align="center",
                    className="d-flex flex-wrap align-items-center justify-content-end pe-3", width=2),
                dbc.Col(id="q_variable",
                        children=[
                            dcc.Dropdown(id="text_var", options=sorted(model.varnames), value=sorted(model.varnames),
                                         multi=True, disabled=False)],
                        width=4, className="d-grid gap-3 border-start border-secondary border-3 rounded-4"),
                dbc.Col(id="e_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_e', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid gap-3 border-start border-3 border-secondary ps-3"),
                dbc.Col(id="e_input",
                        children=[dcc.Dropdown(id={'type': 'i_e', 'index': 0}, disabled=True)], width=3,
                        className="d-grid gap-3 border-end border-secondary border-3 rounded-4")
            ]
        ),
        dbc.Row(dbc.Button("=", id="erg_b", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
        dbc.Row(dbc.Col(html.H2("", className='text-center mb-4', id="head_erg"), className="pt-3", width=12)),
        dbc.Row(
            [
                dbc.Col(dbc.Button("<", id="b_erg_pre", n_clicks=0, disabled=True),
                        className="d-flex justify-content-end align-self-stretch"),
                dbc.Col(children=[], id="pos_erg", className=""),
                dbc.Col(dbc.Button(">", id="b_erg_next", n_clicks=0, disabled=True),
                        className="d-flex justify-content-start align-self-stretch")
            ], className="pt-3", align="center"),
        dbc.Row()

    ], fluid=True
)


@app.callback(
    Output('e_variable', 'children'),
    Output('e_input', 'children'),
    Output('text_l', 'children'),
    Output('q_variable', 'children'),
    Input("upload_tree", 'contents'),
    Input({'type': 'dd_e', 'index': ALL}, 'value'),
    State('e_variable', 'children'),
    State('e_input', 'children'),
    State('q_variable', 'children')
)
def post_router(upload, dd_vals, e_var, e_in, q_var):
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
            return e_var, e_in, c.create_prefix_text_query(len(e_var), len(e_var)), q_var
        e_var_n, e_in_n = c.reset_gui(io_model, "e")
        model = io_model
        priors = model.independent_marginals()
        q_var_n = dcc.Dropdown(id="text_var", options=sorted(model.varnames), value=sorted(model.varnames),
                               multi=True, disabled=False)
        return e_var_n, e_in_n, c.create_prefix_text_query(len(e_var), len(e_var)), [q_var_n]
    elif cb.get("type") == "dd_e":
        if dd_vals[cb.get("index")] is None:
            return c.del_selector_from_div(model, e_var, e_in), c.create_prefix_text_query(4, 4), q_var

        variable = model.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
            minimum = priors[variable].cdf.intervals[0].upper
            maximum = priors[variable].cdf.intervals[-1].lower
            e_in[cb.get("index")] = c.create_range_slider(minimum, maximum,
                                                          id={'type': 'i_e', 'index': cb.get("index")},
                                                          dots=False,
                                                          tooltip={"placement": "bottom", "always_visible": False})

        elif variable.symbolic:
            e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e", "index": cb.get("index")},
                                                 options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                               variable.domain.labels.values())},
                                                 value=list(variable.domain.labels.values()),
                                                 multi=True, )

        if len(e_var) - 1 == cb.get("index"):
            return *c.add_selector_to_div(model, e_var, e_in, "dd_e", cb.get("index") + 1), \
                   c.create_prefix_text_query(len(e_var), len(e_var)), q_var

    return c.update_free_vars_in_div(model, e_var), e_in, c.create_prefix_text_query(len(e_var), len(e_var)), q_var


@app.callback(
    Output('head_erg', 'children'),
    Output('pos_erg', 'children'),
    Output('b_erg_pre', 'disabled'),
    Output('b_erg_next', 'disabled'),
    Input('erg_b', 'n_clicks'),
    Input('b_erg_pre', 'n_clicks'),
    Input('b_erg_next', 'n_clicks'),
    State({'type': 'dd_e', 'index': ALL}, 'value'),
    State({'type': 'i_e', 'index': ALL}, 'value'),
    State('q_variable', 'children'),
)
def erg_controller(n1, n2, n3, e_var, e_in, q_var):
    global result
    global page
    vals = q_var[0]['props']['value']
    cb = ctx.triggered_id
    if cb == "b_erg_pre":
        page -= 1
        if page == 0:
            return vals[page], plot_post(vals, page), True, False
        else:
            return vals[page], plot_post(vals, page), False, False
    elif cb == "b_erg_next":
        page += 1
        if len(vals) > page + 1:
            return vals[page], plot_post(vals, page), False, False
        else:
            return vals[page], plot_post(vals, page), False, True
    else:
        page = 0
        evidence_dict = c.div_to_variablemap(model, e_var, e_in)
        try:
            result = model.posterior(evidence=evidence_dict)
            print("RESULT", result.distributions)
        except Exception as e:
            print("Error was", type(e), e)
            return "", [html.Div("Unsatisfiable", className="fs-1 text text-center pt-3 ")], True, True
        if len(vals) >= 1:
            return vals[page], plot_post(vals, page), True, False
        else:
            return vals[page], plot_post(vals, page), True, True



def plot_post(vars, n):
    var_name = vars[n]
    variable = model.varnames[var_name]
    if variable.numeric:

        return c.plot_numeric_to_div(var_name, result=result)

    elif variable.symbolic:
        return c.plot_symbolic_to_div(var_name, result=result)


if __name__ == '__main__':
    app.run_server(debug=True)

# print(model.variables)
# # model.plot(directory="/tmp/skoomer", view=False, plotvars=model.variables)
# evidence = {model.varnames["x_56"]: [0, 10]}
# evidence = jpt.variables.VariableMap(evidence.items())
# result = model.posterior(evidence=evidence)
# t = plot_numeric_pdf(result["x_56"])

# fig = go.Figure()
# fig.add_trace(t)
# fig.show()

