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
            dbc.Col(html.H1("Most Probable Explanation", className='text-center mb-4'), width=12)
        ),
        dbc.Row(
            [
                dbc.Col([
                    html.Div("argmax ", className="pe-3",
                             style={'fontSize': 20, 'padding-top': 0}),
                    html.Div("P ", className="ps-3",
                             style={'fontSize': 30, 'padding-top': 0}),
                    #html.Div("(", className="text-end float-end align-top pt-0",
                    #         style={"width": "40%", "height": "100%", 'fontSize': 40})
                ], id="text_l", align="center", className="d-flex flex-wrap align-items-center justify-content-end pe-3", width=2),
                dbc.Col(id="q_variable",
                        children=[
                            dcc.Dropdown(id="text_var", options=sorted(model.varnames), value=sorted(model.varnames),
                                         multi=True, disabled=True)],
                        width=4, className="d-grid gap-3 border-start border-secondary border-3 rounded-4"),
                dbc.Col(id="e_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_e', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid gap-3 border-start border-3 border-secondary ps-3"),
                dbc.Col(id="e_input",
                        children=[dcc.Dropdown(id={'type': 'i_e', 'index': 0}, disabled=True)], width=3,
                        className="d-grid gap-3 border-end border-secondary border-3 rounded-4"),
                #dbc.Col(html.Div(")", className="text text-start align-self-center float-start",
                #                 style={"width": "50%", "height": "100%", 'fontSize': 40}), id="text_r")
            ]
        ),
        dbc.Row(dbc.Button("=", id="erg_b", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
        dbc.Row(
            [
                dbc.Col(dbc.Button("<", id="b_erg_pre", n_clicks=0, disabled=True),  className="d-flex justify-content-end align-self-stretch"),
                dbc.Col(children=[], id="mpe_erg", className=""),
                dbc.Col(dbc.Button(">", id="b_erg_next", n_clicks=0, disabled=True), className="d-flex justify-content-start align-self-stretch")
            ], className="pt-3", align="center"),
        dbc.Row()

    ], fluid=True
)

def create_range_slider(variable, *args, **kwargs):
    min = priors[variable].cdf.intervals[0].upper
    max = priors[variable].cdf.intervals[-1].lower

    if min == max:
        min = min - 1
        max = max + 1

    slider = dcc.RangeSlider(**kwargs, min=min, max=max, allowCross=False,)

    return slider



@app.callback(
    Output('e_variable', 'children'),
    Output('e_input', 'children'),

    #Output('text_r', 'children'),
    Output('text_l', 'children'),


    Input({'type': 'dd_e', 'index': ALL}, 'value'),
    State('e_variable', 'children'),
    State('e_input', 'children'),
)
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
            text_l = [
                html.Div("argmax ", className="pe-3",
                         style={"width": "30%", 'fontSize': len(e_var) * 10 if len(e_var)*10 < 30 else 30}),
                html.Div("P ", className="ps-3",
                         style={"width": "30%", "height": "100%", 'fontSize': len(e_var) * 15 if len(e_var) *15 < 75 else 75}),
                #html.Div("(", className="text-end float-end align-top pt-0",
                #         style={"width": "40%", "height": "100%", 'fontSize': len(e_var) * 40 if len(e_var) * 40 < 360 else 360})
            ]
            text_r = [html.Div(")", className="text text-start align-self-center float-start",
                               style={"width": "50%", "height": "100%",
                                      'fontSize':  len(e_var) * 40 if len(e_var) * 40 < 360 else 360,  "padding-top": (len(e_var)-9) * 15 if len(e_var) > 9 else 0})]

            return e_var, e_in, text_l

        variable = model.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
            e_in[cb.get("index")] = create_range_slider(variable, id={'type': 'i_e', 'index': cb.get("index")},
                                                        dots=False,
                                                        tooltip={"placement": "bottom", "always_visible": False})

        elif variable.symbolic:
            e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e", "index": cb.get("index")},
                                                 options={k: v for k, v in zip(variable.domain.labels.keys(),
                                                                               variable.domain.labels.values())},
                                                 multi=True, )

        if len(e_var) - 1 == cb.get("index"):
            e_var.append(
                dcc.Dropdown(id={'type': 'dd_e', 'index': cb.get("index") + 1}, options=sorted(model.varnames)))
            e_in.append(dcc.Dropdown(id={'type': 'i_e', 'index': cb.get("index") + 1}, disabled=True))

    text_l = [
        html.Div("argmax ", className="pe-3",
                 style={"width": "30%", 'fontSize': len(e_var) * 10 if len(e_var) * 10 < 30 else 30}),
        html.Div("P ", className="ps-3",
                 style={"width": "30%", "height": "100%", 'fontSize': len(e_var) * 15 if len(e_var) * 15 < 75 else 75}),
        #html.Div("(", className="text-end float-end align-top pt-0",
        #         style={"width": "40%", "height": "100%", 'fontSize': len(e_var) * 40 if len(e_var) * 40 < 360 else 360})
    ]
    text_r = [html.Div(")", className="text text-start align-self-center float-start",
                       style={"width": "50%", "height": "100%",
                              'fontSize': len(e_var) * 40 if len(e_var) * 40 < 360 else 360, "padding-top": (len(e_var)-9) * 15 if len(e_var) > 9 else 0})]

    return e_var, e_in, text_l



@app.callback(
    Output('mpe_erg', 'children'),
    Output('b_erg_pre','disabled'),
    Output('b_erg_next', 'disabled'),
    Input('erg_b', 'n_clicks'),
    Input('b_erg_pre', 'n_clicks'),
    Input('b_erg_next', 'n_clicks'),
    State({'type': 'dd_e', 'index': ALL}, 'value'),
    State({'type': 'i_e', 'index': ALL}, 'value'),
)
def erg_controller(n1, n2, n3, e_var, e_in):
    global result
    global page
    cb = ctx.triggered_id
    if cb == "b_erg_pre":
        page -= 1
        if page == 0:
            return mpe(result[page]), True, False
        else:
            return mpe(result[page]), False, False
    elif cb == "b_erg_next":
        page += 1
        if len(result) > page+1:
            return mpe(result[page]), False, False
        else:
            return mpe(result[page]), False, True
    else:
        page = 0
        evidence_dict = {}
        for j in range(0, len(e_var) - 1):
            variable = model.varnames[e_var[j]]
            print(variable.domain.labels)
            if variable.numeric:
                evidence_dict.update({e_var[j]: e_in[j]})
            else:
                evidence_dict.update({e_var[j]: set(e_in[j])})
        try:
            result = model._mpe(evidence=evidence_dict)

        except:
            return [html.Div("Unsatisfiable", className="fs-1 text text-center pt-3 ")], True, True
        if len(result) > 1:
            return mpe(result[0]), True, False
        else:
            return mpe(result[0]), True, True



def mpe(res):

    return_div = []
    for variable, restriction in res.items():

        if variable.numeric:
            value = []
            for interval in res[variable]:
                value += [interval.lower, interval.upper]

            return_div += [html.Div(
                [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True, className="margin10"),
                 create_range_slider(variable, value=value, disabled=True, className="margin10",  tooltip={"placement": "bottom", "always_visible": True})]
                , style={"display": "grid", "grid-template-columns": "30% 70%"})]
        elif variable.symbolic:
            return_div += [html.Div(
                [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True),
                 dcc.Dropdown(
                     options={k: v for k, v in zip(variable.domain.labels.keys(), variable.domain.labels.values())},
                     value=restriction, multi=True, disabled=True, className="ps-3")],
                style={"display": "grid", "grid-template-columns": "30% 70%"})]

    return_div += [html.Div(className="pt-3")]
    return return_div


if __name__ == '__main__':

    app.run_server(debug=True)
