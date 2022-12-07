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
from typing import List

global model
model: jpt.trees.JPT = c.default_tree

global priors
priors = model.independent_marginals()

global modal_var_index
modal_var_index = -1

global modal_type
# 0 = q and 1 = e
modal_type = -1

global modal_basic
modal_basic = c.modal_basic

modal_option = c.modal_option

app = dash.Dash(__name__, prevent_initial_callbacks=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H1("Query", className='text-center mb-4'), width=12),
                dbc.Col(dcc.Upload(children=dbc.Button("🌱", n_clicks=0, className="position-absolute top-0 end-0"),
                                   id="upload_tree"))
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
                        className="d-grid gap-3 "),
                dbc.Col(children=[html.Div(id="q_option", children=[
                    dbc.Button("👁️", id=dict(type='b_q', index=0), disabled=True, n_clicks=0, className="me-2 mb-3",
                               size="sm")], className=" d-grid align-self-start")
                                  ],
                        className="d-grid gap-0 gx-0 d-flex align-items-stretch flex-grow-0 align-self-stretch border-end border-3 border-secondary "),
                dbc.Col(id="e_variable",
                        children=[dcc.Dropdown(id={'type': 'dd_e', 'index': 0}, options=sorted(model.varnames))],
                        width=1, className="d-grid gap-0 border-start border-3 border-secondary ps-3"),
                dbc.Col(id="e_input",
                        children=[dcc.Dropdown(id={'type': 'i_e', 'index': 0}, disabled=True)], width=3,
                        className="d-grid gap-3 "),
                dbc.Col(children=[html.Div(id="e_option", children=[
                    dbc.Button("👁️", id=dict(type='b_e', index=0), disabled=True, n_clicks=0, className="me-2 mb-3",
                               size="sm")],
                                           className=" d-grid border-end border-secondary border-3 rounded-4")
                                  ],
                        className="d-grid gx-1 d-md-flex align-self-center"),
            ], className="justify-content-center",
        ),
        dbc.Row(dbc.Button("=", id="erg_b", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
        dbc.Row(dbc.Col(html.Div("", id="erg_text", className="fs-1 text text-center pt-3 "))),
        modal_option
    ], fluid=True
)


def query_gen(dd_vals: List, q_var: List, q_in: List, q_op):
    """
    Handel all action in the Query Part of the GUI (Extend Change Reduce)
    :param dd_vals: All Varietals used in Query Section are chosen
    :param q_var: the Dropdown of variable of Query Section
    :param q_in: the Input for the Variables of Query Section
    :return: Updatet Varibel List and the Input.
    """
    cb = ctx.triggered_id
    if dd_vals[cb.get("index")] is None:
        return c.del_selector_from_div_button(model, q_var, q_in, q_op, cb.get("index"))

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
        return c.add_selector_to_div_button(model, q_var, q_in, q_op, 'q', cb.get("index") + 1)
    return c.update_free_vars_in_div(model, q_var), q_in, q_op


def evid_gen(dd_vals, e_var, e_in, e_op):
    """
    Handel all action in the Evidence Part of the GUI (Extend Change Reduce)
    :param dd_vals: All Varietals used in Evidence Section are chosen
    :param e_var: the Dropdown of variable of Evidence Section
    :param e_in: the Input for the Variables of Evidence Section
    :return: Updatet Varibel List and the Input.
    """
    e_var: List[dict] = e_var
    e_in: List[dict] = e_in
    cb = ctx.triggered_id
    print(cb)
    if dd_vals[cb.get("index")] is None:
        return c.del_selector_from_div_button(model, e_var, e_in, e_op, cb.get('index'))

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
        return c.add_selector_to_div_button(model, e_var, e_in, e_op, "e", cb.get("index") + 1)
    return c.update_free_vars_in_div(model, e_var), e_in, e_op


@app.callback(
    Output('q_variable', 'children'),
    Output('q_input', 'children'),
    Output('q_option', 'children'),
    Output('e_variable', 'children'),
    Output('e_input', 'children'),
    Output('e_option', 'children'),
    Output('text_l', 'children'),
    Output('modal_option', 'children'),
    Output('modal_option', 'is_open'),
    Input("upload_tree", "contents"),
    Input({'type': 'dd_q', 'index': ALL}, 'value'),
    Input({'type': 'dd_e', 'index': ALL}, 'value'),
    Input({'type': 'b_q', 'index': ALL}, 'n_clicks'),
    Input({'type': 'b_e', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_save', 'index': ALL}, 'n_clicks'),
    State('q_variable', 'children'),
    State('q_input', 'children'),
    State('e_variable', 'children'),
    State('e_input', 'children'),
    State('q_option', 'children'),
    State('e_option', 'children'),
    State({'type': 'op_i', 'index': ALL}, 'value'),
)
def query_router(upload, q_dd, e_dd, b_q, b_e, op_s, q_var, q_in, e_var, e_in, q_op, e_op, op_i):
    """
    Receives app callback events and manages/redirects these to the correct functions.
    :param upload: Path to the new jpt Tree as a File
    :param q_dd: Query Varibels Names
    :param e_dd: Evidence Variable Names
    :param q_var: Div of the Query Variable
    :param q_in: Div or the Input of Query
    :param e_var: Div of the Evidence Variable
    :param e_in: Div or the Input of Evidence
    :return: Query Varibels, Query Input, Evidence Variable, Evidence Input, Text Prefix.
    """
    global modal_var_index
    global modal_type  # 0 = q and 1 = e

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
            return q_var, q_in, q_op, e_var, e_in, e_op, \
                   c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic, False

        model = io_model
        priors = model.independent_marginals()
        return *c.reset_gui_button(io_model, "q"), *c.reset_gui_button(io_model, "e"), \
               c.create_prefix_text_query(len_fac_q=2, len_fac_e=2), modal_basic, False
    elif cb.get("type") == "dd_q":
        return *query_gen(q_dd, q_var, q_in, q_op), e_var, e_in, e_op, \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic, False
    elif cb.get("type") == "dd_e":

        return q_var, q_in, q_op, *evid_gen(e_dd, e_var, e_in, e_op), \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic, False
    elif cb.get("type") == "b_e" and e_dd[cb.get("index")] != []:

        modal_var_index = cb.get("index")
        modal_type = 1
        #Wenn kein Value exestiert wird min max genommen
        modal_body = c.generate_modal_option(model=model, var=e_var[cb.get("index")]['props']['value'],
                                             value=e_in[cb.get("index")]['props']
                                             .get('value', [e_in[cb.get("index")]['props']['min'],
                                                           e_in[cb.get("index")]['props']['max']]),
                                             priors=priors)
        return q_var, q_in, q_op, e_var, e_in, e_op, \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_body, True
    elif cb.get("type") == "b_q" and q_dd[cb.get("index")] != []:

        modal_var_index = cb.get("index")
        modal_type = 0


        modal_body = c.generate_modal_option(model=model, var=q_var[cb.get("index")]['props']['value'],
                                             value=q_in[cb.get("index")]['props'].get('value',
                                                                                      [q_in[cb.get("index")]['props']['min'], q_in[cb.get("index")]['props']['max']]), priors=priors)
        return q_var, q_in, q_op, e_var, e_in, e_op, \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_body, True
    elif cb.get("type") == "option_save":
        new_vals = c.fuse_overlapping_range(op_i)
        if modal_type == 1:
            e_in[modal_var_index]['props']['value'] = new_vals
            e_in[modal_var_index]['props']['drag_value'] = new_vals
            return q_var, q_in, q_op, e_var, e_in, e_op, \
                   c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic, False
        else:
            q_in[modal_var_index]['props']['value'] = new_vals
            q_in[modal_var_index]['props']['drag_value'] = new_vals
            return q_var, q_in, q_op, e_var, e_in, e_op, \
                   c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic, False
    else:
        return q_var, q_in, q_op, e_var, e_in, e_op, \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic, False


@app.callback(
    Output("modal_input", "children"),
    Input("op_add", "n_clicks"),
    Input({'type': 'op_i', 'index': ALL}, 'value'),
    State("modal_input", "children"),
    State({'type': 'dd_e', 'index': ALL}, 'value'),
    State({'type': 'dd_q', 'index': ALL}, 'value'),
)
def modal_router(op, op_i, m_bod, dd_e, dd_q):
    cb = ctx.triggered_id
    global modal_var_index
    global modal_type
    dd = dd_e if modal_type == 1 else dd_q
    var = dd[modal_var_index]
    if not isinstance(m_bod, list):
        m_in_new = [m_bod]
    else:
        m_in_new = m_bod
    if cb == "op_add":
        index = m_in_new[-2]['props']['children'][0]['props']['children'][1]['props']['id']['index']
        type = m_in_new[1]['props']['children'][0]['props']['children'][1]['type']
        if type == "RangeSlider":

            mini = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['min']
            maxi = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['max']
            range_string = html.Div(f"Range {index + 2}",
                                    style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal)-1)]))
            n_slider = c.create_range_slider(minimum=mini, maximum=maxi,id={'type': 'op_i', 'index': index + 1},
                                             value=[mini, maxi], dots=False,
                                             tooltip={"placement": "bottom", "always_visible": False},
                                             className="flex-fill")
            var_map = c.div_to_variablemap(model, [var], [[mini, maxi]])
            prob = model.infer(var_map, {})

            prob_div = html.Div(f"{prob}", style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal)-1)]))
            m_in_new.insert(len(m_in_new) - 1, dbc.Row([
                html.Div([range_string, n_slider, prob_div], id=f"modal_color_{(index + 1) % (len(c.color_list_modal)-1)}", className="d-flex flex-nowrap justify-content-center ps-2")
            ],className="d-flex justify-content-center"))
            return m_in_new
        else:
            # Sollte nicht Triggerbar sein, da bei DDMenu der +Buttone nicht Aktiv ist
            return m_in_new
    else:  # if cb.get("type") == "op_i"
        id = cb.get("index")
        value = m_in_new[id + 1]['props']['children'][0]['props']['children'][1]['props']['value']
        var_map = c.div_to_variablemap(model, [var], [value])
        prob = model.infer(var_map, {})
        prob_div = html.Div(f"{prob}", style=dict(color=c.color_list_modal[id % (len(c.color_list_modal)-1)]))
        m_in_new[id + 1]['props']['children'][0]['props']['children'][2] = prob_div
        return m_in_new


@app.callback(
    Output("erg_text", "children"),
    Input("erg_b", "n_clicks"),

    State({'type': 'dd_q', 'index': ALL}, 'value'),
    State({'type': 'i_q', 'index': ALL}, 'value'),
    State({'type': 'dd_e', 'index': ALL}, 'value'),
    State({'type': 'i_e', 'index': ALL}, 'value'),
)
def infer(n1, q_var, q_in, e_var, e_in):
    """
    Calculates withe Jpt the Probilty of query and evidence
    :param n1: Button to trigger the Calculation
    :param q_var: Div of the Query Variable
    :param q_in: Div or the Input of Query
    :param e_var: Div of the Evidence Variable
    :param e_in: Div or the Input of Evidence
    :return: Probability as String
    """
    query = c.div_to_variablemap(model, q_var, q_in)
    evidence = c.div_to_variablemap(model, e_var, e_in)
    print(f'query:{query}, evi:{evidence}')
    try:
        result = model.infer(query, evidence)

    except Exception as e:
        print(e)
        return "Unsatasfiable"
    print(result)
    return "{}%".format(round(result.result * 100, 2))


if __name__ == '__main__':
    app.run_server(debug=True)

# DUBILCA VERBIERTEN

# 2. Posterior RESULTS
# 3. MASKE DTAILS GUI
# 4. MEHER SLIDER ODER IN MPE UND QUERY
# 5. https://observablehq.com/@d3/tree-of-life
# 6. ERKLÄREN QUERY Button EXPLAN
# PIP
# JPT Update
# Landing page
