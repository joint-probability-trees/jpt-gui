import base64

import jpt
import igraph
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
model: jpt.trees.JPT = jpt.JPT.load('test.datei')

global priors
priors = model.independent_marginals()

app = dash.Dash(__name__, prevent_initial_callbacks=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H1("Query", className='text-center mb-4'), width=12),
                dbc.Col(dcc.Upload(children=dbc.Button("🌱", n_clicks=0, className="position-absolute top-0 end-0"), id="upload_tree"))
            ]
        ),
        dbc.Row(
            [
                dbc.Col([
                    html.Div("P ", className="align-self-center text-end float-end",
                             style={'fontSize': 40, 'padding-top': 0}),
                ], id="text_l", align="center", className="", width=2),
                dbc.Col(id="q_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_q', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid gap-3 border-start border-secondary border-3 rounded-4"),
                dbc.Col(id="q_input",
                        children=[dcc.Dropdown(id={'type': 'i_q', 'index': 0}, disabled=True)], width=3,
                        className="d-grid gap-3 border-end border-3 border-secondary"),
                dbc.Col(id="e_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_e', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid gap-3 border-start border-3 border-secondary ps-3"),
                dbc.Col(id="e_input",
                        children=[dcc.Dropdown(id={'type': 'i_e', 'index': 0}, disabled=True)], width=3,
                        className="d-grid gap-3 border-end border-secondary border-3 rounded-4"),
            ]
        ),
        dbc.Row(dbc.Button("=", id="erg_b", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
        dbc.Row(dbc.Col(html.Div("", id="erg_text", className="fs-1 text text-center pt-3 ")))

    ], fluid=True
)


def query_gen(dd_vals, q_var, q_in):
    q_var: list[dict] = q_var
    q_in: list[dict] = q_in

    cb = ctx.triggered_id
    if dd_vals[cb.get("index")] is None:
        return c.del_selector_from_div(model, q_var, q_in, cb.get("index"))

    variable = model.varnames[dd_vals[cb.get("index")]]
    if variable.numeric:
        minimum = priors[variable].cdf.intervals[0].upper
        maximum = priors[variable].cdf.intervals[-1].lower
        q_in[cb.get("index")] = c.create_range_slider(minimum, maximum, id={'type': 'i_q', 'index': cb.get("index")},
                                                      tooltip={"placement": "bottom", "always_visible": False})

    elif variable.symbolic:
        q_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_q", "index": cb.get("index")},
                                             options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                           variable.domain.labels.values())},
                                             value=list(variable.domain.labels.values()),
                                             multi=True, )  # list(variable.domain.labels.keys())

    if len(q_var) - 1 == cb.get("index"):
        return c.add_selector_to_div(model, q_var, q_in, 'dd_q', cb.get("index")+1)


    return c.update_free_vars_in_div(model, q_var), q_in


def evid_gen(dd_vals, e_var, e_in):
    e_var: list[dict] = e_var
    e_in: list[dict] = e_in
    cb = ctx.triggered_id
    print(cb)
    if dd_vals[cb.get("index")] is None:
        return c.del_selector_from_div(model, e_var, e_in, cb.get('index'))

    variable = model.varnames[dd_vals[cb.get("index")]]
    if variable.numeric:
        minimum = priors[variable].cdf.intervals[0].upper
        maximum = priors[variable].cdf.intervals[-1].lower
        e_in[cb.get("index")] = c.create_range_slider(minimum, maximum, id={'type': 'i_e', 'index': cb.get("index")},
                                                      tooltip={"placement": "bottom", "always_visible": False})
    elif variable.symbolic:
        e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e", "index": cb.get("index")},
                                             options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                           variable.domain.labels.values())},
                                             value=list(variable.domain.labels.values()), multi=True, )

    if len(e_var) - 1 == cb.get("index"):
        return c.add_selector_to_div(model,e_var, e_in, "dd_e", cb.get("index")+1)


    return c.update_free_vars_in_div(model, e_var), e_in


@app.callback(
    Output('q_variable', 'children'),
    Output('q_input', 'children'),

    Output('e_variable', 'children'),
    Output('e_input', 'children'),

    Output('text_l', 'children'),

    Input("upload_tree", "contents"),

    Input({'type': 'dd_q', 'index': ALL}, 'value'),
    Input({'type': 'dd_e', 'index': ALL}, 'value'),

    State('q_variable', 'children'),
    State('q_input', 'children'),

    State('e_variable', 'children'),
    State('e_input', 'children'),
)
def query_router(upload, q_dd, e_dd, q_var, q_in, e_var, e_in):
    cb = ctx.triggered_id
    print(cb)
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
                return q_var, q_in, e_var, e_in, c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var))

            q_var_n, q_in_n = c.reset_gui(io_model, "q")
            e_var_n, e_in_n = c.reset_gui(io_model, "e")
            model = io_model
            priors = model.independent_marginals()
            return q_var_n, q_in_n, e_var_n, e_in_n, c.create_prefix_text_query(len_fac_q=2,
                                                                                len_fac_e=2)
    elif cb.get("type") == "dd_q":
        q_var_n, q_in_n = query_gen(q_dd, q_var, q_in)
        print(q_var_n, q_in_n)
        return q_var_n, q_in_n, e_var, e_in, c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var))
    elif cb.get("type") == "dd_e":
        e_var_n, e_in_n = evid_gen(e_dd, e_var, e_in)
        return q_var, q_in, e_var_n, e_in_n, c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var))
    else:
        return q_var, q_in, e_var, e_in, c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var))


@app.callback(
    Output("erg_text", "children"),
    Input("erg_b", "n_clicks"),

    State({'type': 'dd_q', 'index': ALL}, 'value'),
    State({'type': 'i_q', 'index': ALL}, 'value'),
    State({'type': 'dd_e', 'index': ALL}, 'value'),
    State({'type': 'i_e', 'index': ALL}, 'value'),
)
def infer(n1, q_var, q_in, e_var, e_in):
    query = c.div_to_variablemap(model, q_var, q_in)
    evidence = c.div_to_variablemap(model, e_var, e_in)
    print(f'qery:{query}, evi:{evidence}')
    try:
        result = model.infer(query, evidence)
    except Exception as e:
        print(e)
        return "Unsatasfiable"
    print(result)
    return "{}%".format(round(result.result * 100, 2))


if __name__ == '__main__':
    app.run_server(debug=True)

#DUBILCA VERBIERTEN

#1. VAR DUBLKICA VERBIERTEN + STYLE MPE ÄNDERN
#1.5 UPLOAD JPT BAUM
#2. Posterior RESULTS
#3. MASKE DTAILS GUI
#4. ERKLÄREN QUERY Button EXPLAN
#5. MEHER SLIDER ODER IN MPE UND QUERY
#6. https://observablehq.com/@d3/tree-of-life
