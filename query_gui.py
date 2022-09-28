import jpt
import igraph
from igraph import Graph, EdgeSeq
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from jpt.base.utils import list2interval
import dash
from dash import dcc, html, Input, Output, State, ctx, MATCH, ALLSMALLER, ALL
import math

global model
model: jpt.trees.JPT = jpt.JPT.load('test.datei')

global priors
priors = model.independent_marginals()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], prevent_initial_callbacks=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(html.H1("Query", className='text-center mb-4'), width=12)
        ),
        dbc.Row(
            [
                dbc.Col([
                         html.Div("P ", className="align-self-center text-end float-start", style={"width": "50%", 'fontSize': 40, 'padding-top': 0}),
                         html.Div("(", className="text-end float-end align-top pt-0", style={"width": "50%", "height": "100%", 'fontSize': 40})
                         ], id="text_l", align="center"),
                dbc.Col(id="q_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_q', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid gap-3"),
                dbc.Col(id="q_input",
                        children=[dcc.Dropdown(id={'type': 'i_q', 'index': 0}, disabled=True)], width=3, className="d-grid gap-3 border-end border-3 border-secondary border-start-3 rounded-5"),
                #dbc.Col(html.Div("|", className="fs-1 text text-center")),
                dbc.Col(id="e_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_e', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid gap-3 border-start border-3 border-secondary ps-3"),
                dbc.Col(id="e_input",
                        children=[dcc.Dropdown(id={'type': 'i_e', 'index': 0}, disabled=True)], width=3, className="d-grid gap-3"),
                dbc.Col(html.Div(")", className="text text-start align-self-center float-start", style={"width": "50%", "height": "100%", 'fontSize': 40}), id="text_r")
            ]
        ),
        dbc.Row(dbc.Button("=", id="erg_b", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
        dbc.Row(dbc.Col(html.Div("", id="erg_text", className="fs-1 text text-center pt-3 ")))

    ], fluid=True
)


# @app.callback(
#     Output('q_variable', 'children'),
#     Output('q_input', 'children'),
#     Output('text_l', 'children'),
#     Output('text_r', 'children'),
#     Input({'type': 'dd_q', 'index': ALL}, 'value'),
#     State('q_variable', 'children'),
#     State('q_input', 'children'),
# )
def query_gen(dd_vals, q_var, q_in):
    q_var: list[dict] = q_var
    q_in: list[dict] = q_in

    cb = ctx.triggered_id
    print(cb)
    if (cb.get("type") == "dd_q"):
        if dd_vals[cb.get("index")] is None:
            q_var.pop(cb.get("index"))
            q_in.pop(cb.get("index"))
            for x in range(0, len(q_var)):
                q_var[x]['props']['id'] = {'type': 'dd_q', 'index': x}
                q_in[x]['props']['id'] = {'type': 'i_q', 'index': x}
            return q_var, q_in

        variable = model.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
                min = priors[variable].cdf.intervals[0].upper
                max = priors[variable].cdf.intervals[-1].lower

                if min == max:
                    min = min - 1
                    max = max + 1

                q_in[cb.get("index")] = dcc.RangeSlider(id={'type': 'i_q', 'index': cb.get("index")},
                                                        min=min, max=max, value=[min, max],
                                                        allowCross=False,
                                                        tooltip={"placement": "bottom", "always_visible": False})
        elif variable.symbolic:
            q_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_q", "index": cb.get("index")},
                                                 options={k:v for k,v in zip(variable.domain.labels.keys(), variable.domain.labels.values()) },value=list(variable.domain.labels.keys()), multi=True, ) #list(variable.domain.labels.keys())

        if len(q_var) - 1 == cb.get("index"):
            q_var.append(
                dcc.Dropdown(id={'type': 'dd_q', 'index': cb.get("index") + 1}, options=sorted(model.varnames)))
            q_in.append(dcc.Dropdown(id={'type': 'i_q', 'index': cb.get("index") + 1}, disabled=True))


        return q_var, q_in
    return q_var, q_in


# @app.callback(
#     Output('e_variable', 'children'),
#     Output('e_input', 'children'),
#
#     Input({'type': 'dd_e', 'index': ALL}, 'value'),
#     State('e_variable', 'children'),
#     State('e_input', 'children'),
# )
def evid_gen(dd_vals, e_var, e_in):
    e_var: list[dict] = e_var
    e_in: list[dict] = e_in
    cb = ctx.triggered_id
    print(cb)
    if (cb.get("type") == "dd_e"):
        if dd_vals[cb.get("index")] is None:
            e_var.pop(cb.get("index"))
            e_in.pop(cb.get("index"))
            for x in range(0, len(e_var)):
                e_var[x]['props']['id'] = {'type': 'dd_e', 'index': x}
                e_in[x]['props']['id'] = {'type': 'i_e', 'index': x}
            return e_var, e_in

        variable = model.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
            # expectation = model.expectation([variable], {}, confidence_level=1.)
            # expectation = model.expectation([variable], {}, confidence_level=1.)

                min = priors[variable].cdf.intervals[0].upper
                max = priors[variable].cdf.intervals[-1].lower

                if min == max:
                    min = min - 1
                    max = max + 1

                e_in[cb.get("index")] = dcc.RangeSlider(id={'type': 'i_e', 'index': cb.get("index")},
                                                        min=min, max=max, value=[min, max],
                                                        allowCross=False,
                                                        tooltip={"placement": "bottom", "always_visible": False})

        elif variable.symbolic:
            e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e", "index": cb.get("index")},
                                                 options={k:v for k,v in zip(variable.domain.labels.keys(), variable.domain.labels.values()) }, multi=True, )

        if len(e_var) - 1 == cb.get("index"):
            e_var.append(
                dcc.Dropdown(id={'type': 'dd_e', 'index': cb.get("index") + 1}, options=sorted(model.varnames)))
            e_in.append(dcc.Dropdown(id={'type': 'i_e', 'index': cb.get("index") + 1}, disabled=True))
        return e_var, e_in
    return e_var, e_in

@app.callback(
    Output('q_variable', 'children'),
    Output('q_input', 'children'),

    Output('e_variable', 'children'),
    Output('e_input', 'children'),

    Output('text_r', 'children'),
    Output('text_l', 'children'),


    Input({'type': 'dd_q', 'index': ALL}, 'value'),
    Input({'type': 'dd_e', 'index': ALL}, 'value'),

    State('q_variable', 'children'),
    State('q_input', 'children'),

    State('e_variable', 'children'),
    State('e_input', 'children'),
)
def query_router(q_dd, e_dd, q_var, q_in, e_var, e_in):
    cb = ctx.triggered_id
    print(cb)
    if cb.get("type") == "dd_q":
        q_var_n, q_in_n = query_gen(q_dd, q_var, q_in)
        text_r = [
            html.Div("P ", className="align-self-center text-end float-start",
                     style={"width": "50%", "height": "100%", 'fontSize': (len(q_var) if len(q_var) >= len(e_var) else len(e_var))*20,  'padding-top': (len(q_var) if len(q_var) >= len(e_var) else len(e_var))*20}),
            html.Div("(", className="text-end float-end align-top pt-0",
                     style={"width": "50%", "height": "100%", 'fontSize': (len(q_var_n) if len(q_var_n) >= len(e_var) else len(e_var)) * 40})
        ]
        text_l = [html.Div(")", className="text text-start align-self-center float-start",
                           style={"width": "50%", "height": "100%", 'fontSize': (len(q_var_n) if len(q_var_n) >= len(e_var) else len(e_var)) * 40})]
        return q_var_n, q_in_n, e_var, e_in, text_l, text_r
    elif cb.get("type") == "dd_e":
        e_var_n, e_in_n = evid_gen(e_dd,e_var,e_in)
        text_r = [
            html.Div("P ", className="align-self-center text-end float-start",
                     style={"width": "50%", "height": "100%", 'fontSize': (len(q_var) if len(q_var) >= len(e_var) else len(e_var))*20,  'padding-top': (len(q_var) if len(q_var) >= len(e_var) else len(e_var))*20}),
            html.Div("(", className="text-end float-end align-top pt-0",
                     style={"width": "50%", "height": "100%", 'fontSize': (len(q_var) if len(q_var) >= len(e_var_n) else len(e_var_n)) * 40})
        ]
        text_l = [html.Div(")", className="text text-start align-self-center float-start",
                           style={"width": "50%", "height": "100%", 'fontSize': (len(q_var) if len(q_var) >= len(e_var_n) else len(e_var_n)) * 40})]
        return q_var, q_in, e_var_n, e_in_n, text_l, text_r
    else:
        text_r = [
            html.Div("P ", className="align-self-center text-end float-start",
                     style={"width": "50%", "height": "100%", 'fontSize': (len(q_var) if len(q_var) >= len(e_var) else len(e_var))*20,  'padding-top': (len(q_var) if len(q_var) >= len(e_var) else len(e_var))*20}),
            html.Div("(", className="text-end float-end align-top pt-0",
                     style={"width": "50%", "height": "100%", 'fontSize': (len(q_var) if len(q_var) >= len(e_var) else len(e_var)) * 40})
        ]
        text_l = [html.Div(")", className="text text-start align-self-center float-start",
                           style={"width": "50%", "height": "100%", 'fontSize': (len(q_var) if len(q_var) >= len(e_var) else len(e_var)) * 40})]
        return q_var, q_in, e_var, e_in, text_l, text_r



@app.callback(
    Output("erg_text", "children"),
    Input("erg_b", "n_clicks"),

    State({'type': 'dd_q', 'index': ALL}, 'value'),
    State({'type': 'i_q', 'index': ALL}, 'value'),
    State({'type': 'dd_e', 'index': ALL}, 'value'),
    State({'type': 'i_e', 'index': ALL}, 'value'),
)
def infer(n1, q_var, q_in, e_var, e_in):
    query_dict = {}
    evidence_dict = {}
    for i in range(0, len(q_var) - 1):
        variable = model.varnames[q_var[i]]
        print(variable.domain.labels)
        if variable.numeric:
            query_dict.update({q_var[i]: q_in[i]})
        else:
            query_dict.update({q_var[i]: set(q_in[i])})

    for j in range(0, len(e_var) - 1):
        variable = model.varnames[e_var[j]]
        print(variable.domain.labels)
        if variable.numeric:
            evidence_dict.update({e_var[j]: e_in[j]})
        else:
            evidence_dict.update({e_var[j]: set(e_in[j])})

    # ToDoo Fors Kombenieren
    evidence = jpt.variables.VariableMap([(model.varnames[k], v) for k, v in evidence_dict.items()])
    query = jpt.variables.VariableMap([(model.varnames[k], v) for k, v in query_dict.items()])
    print(evidence, query)

    try:
        result = model.infer(query, evidence)
    except:
        return "Unsatasfiable"

    return "{}%".format(round(result.result * 100, 2))


if __name__ == '__main__':
    app.run_server(debug=True)
# result.result *100 '%'
