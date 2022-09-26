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

global expectation
expectation = model.expectation(model.variables, evidence=jpt.variables.VariableMap(), confidence_level=1.)



app = dash.Dash(__name__, prevent_initial_callbacks=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(html.H1("MPE", className='text-center mb-4'), width=12)
        ),
        dbc.Row(
            [
                dbc.Col([
                         html.Div("argmax ", className="align-self-center text-end border float-start border-white ", style={"width": "30%", 'fontSize': 20, 'padding-top': 0}),
                         html.Div("P ", className="align-self-center text-end  border border-white", style={"width": "30%", 'fontSize': 30, 'padding-top': 0}),
                         html.Div("(", className="text-end float-end align-top pt-0 border border-white", style={"width": "40%", "height": "100%", 'fontSize': 40})
                         ], id="text_l", align="center", className="d-flex"),
                dbc.Col(id="q_variable",
                        children=[dcc.Dropdown(id="text_var", options=sorted(model.varnames), value=sorted(model.varnames), multi=True, disabled=True)],
                        #children=[html.Div(sorted(model.varnames).__str__(), id="text_var", className="text-warp text-break", style={"text-align": "justify"})],
                        width=4, className="d-grid gap-3"),
                dbc.Col(id="e_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_e', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid gap-3 border-start border-3 border-secondary ps-3"),
                dbc.Col(id="e_input",
                        children=[dcc.Dropdown(id={'type': 'i_e', 'index': 0}, disabled=True)], width=3, className="d-grid gap-3"),
                dbc.Col(html.Div(")", className="text text-start align-self-center float-start", style={"width": "50%", "height": "100%", 'fontSize': 40}), id="text_r")
            ]
        ),
        dbc.Row(dbc.Button("=", id="erg_b", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
        dbc.Row(dbc.Col(children=[], id="mpe_erg", className="d-grid gap-2 col-3 mt-3 mx-auto"))

    ], fluid=True
)

@app.callback(
    Output('e_variable', 'children'),
    Output('e_input', 'children'),

    Output('text_r', 'children'),
    Output('text_l', 'children'),
    Output('text_var', 'value'),

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
                html.Div("argmax ", className="align-self-center text-end border border-white", style={"width": "30%", 'fontSize': 20, 'padding-top': len(e_var)*20}),
                html.Div("P ", className="align-self-center text-end float-start border border-white",
                         style={"width": "30%", "height": "100%", 'fontSize': len(e_var)*15,  'padding-top': len(e_var)*20}),
                html.Div("(", className="text-end float-end align-top pt-0 border border-white",
                         style={"width": "40%", "height": "100%", 'fontSize': len(e_var) * 40})
            ]
            text_r = [html.Div(")", className="text text-start align-self-center float-start",
                               style={"width": "50%", "height": "100%",
                                      'fontSize': len(e_var) * 40})]

            text_var = [x for x in model.varnames if x not in dd_vals]
            return e_var, e_in, text_r, text_l, text_var

        variable = model.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
            # expectation = model.expectation([variable], {}, confidence_level=1.)
            print(expectation[variable])
            if math.isnan(expectation[variable].lower):
                max = math.ceil(expectation[variable].upper)
                e_in[cb.get("index")] = dcc.RangeSlider(id={'type': 'i_e', 'index': cb.get("index")},
                                                        min=max-1, max=max + 1, step=0.1, value=[max-1, max + 1],
                                                        allowCross=False, dots=False,
                                                        tooltip={"placement": "bottom", "always_visible": True})
            elif math.isnan(expectation[variable].upper):
                min = math.ceil(expectation[variable].lower)
                e_in[cb.get("index")] = dcc.RangeSlider(id={'type': 'i_e', 'index': cb.get("index")},
                                                        min=min-1, max=min - 1, step=0.1, value=[min-1, min - 1],
                                                        allowCross=False, dots=False,
                                                        tooltip={"placement": "bottom", "always_visible": True})
            else:
                min = math.ceil(expectation[variable].lower)
                max = math.ceil(expectation[variable].upper)
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

    text_l = [
        html.Div("argmax ", className="align-self-center text-end border border-white",
                 style={"width": "30%", 'fontSize': 20, 'padding-top': len(e_var) * 20}),
        html.Div("P ", className="align-self-center text-end float-start border border-white",
                 style={"width": "30%", "height": "100%", 'fontSize': len(e_var) * 15, 'padding-top': len(e_var) * 20}),
        html.Div("(", className="text-end float-end align-top pt-0 border border-white",
                 style={"width": "40%", "height": "100%", 'fontSize': len(e_var) * 40})
    ]
    text_r = [html.Div(")", className="text text-start align-self-center float-start",
                       style={"width": "50%", "height": "100%",
                              'fontSize': len(e_var) * 40})]

    text_var = [x for x in model.varnames if x not in dd_vals]
    return e_var, e_in, text_r, text_l, text_var


def create_range_slider(variable, *args, **kwargs):
    global expectation

    if math.isnan(expectation[variable].lower):
        max = math.ceil(expectation[variable].upper)
        slider = dcc.RangeSlider(**kwargs,
                                 min=max, max=max + 1, step=0.1, allowCross=False,
                                 dots=False, tooltip={"placement": "bottom", "always_visible": True}
                                 )
    elif math.isnan(expectation[variable].upper):
        min = math.ceil(expectation[variable].lower)
        slider = dcc.RangeSlider(**kwargs, min=min, max=min - 1, step=0.1, allowCross=False, dots=False,
                                 tooltip={"placement": "bottom", "always_visible": True})
    else:
        min = math.ceil(expectation[variable].lower)
        max = math.ceil(expectation[variable].upper)
        slider = dcc.RangeSlider(**kwargs, min=min, max=max, allowCross=False,
                                 tooltip={"placement": "bottom", "always_visible": False})

    return slider


@app.callback(
    Output('mpe_erg', 'children'),
    Input('erg_b', 'n_clicks'),
    State({'type': 'dd_e', 'index': ALL}, 'value'),
    State({'type': 'i_e', 'index': ALL}, 'value'),
)
def mpe(n1, e_var, e_in): #Error bei
    evidence_dict = {}

    for j in range(0, len(e_var) - 1):
        variable = model.varnames[e_var[j]]
        print(variable.domain.labels)
        if variable.numeric:
            evidence_dict.update({e_var[j]: e_in[j]})
        else:
            evidence_dict.update({e_var[j]: set(e_in[j])})
    try:
        result = model.mpe(evidence=evidence_dict)
    except:
        return [html.Div("Unsatasfiable")]
    return_div = []
    for variable, restriction in result.path.items():
        print(type(variable.name), variable.name)
        if variable.numeric:
            print(restriction.lower)
            value = [restriction.lower if restriction.lower > -float("inf") else expectation[variable].lower,
                     restriction.upper if restriction.upper < float("inf") else expectation[variable].upper]
            return_div += [html.Div([dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True, className="margin10"), create_range_slider(variable, value=value, disabled=True, className="margin10" )]
                                    , style={"display": "grid", "grid-template-columns": "30% 70%"})]
            #return_div += [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True, style={"width": "50%"}), create_range_slider(variable, value=value, disabled=True,)]
        elif variable.symbolic:
             #return_div += [html.Div()]
             return_div += [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True, style={"width": "50%"}), dcc.Dropdown(options={k: v for k, v in zip(variable.domain.labels.keys(), variable.domain.labels.values())},
                         value=restriction, multi=True, disabled=True,style={"width": "50%"})]

    return return_div
    # min=excet lower  max= excet upper  l=lower path   u=upper path
    # sym drobdown
if __name__ == '__main__':
    app.run_server(debug=True)