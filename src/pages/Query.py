import dash_bootstrap_components as dbc
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, callback
from src import components as c
from typing import List
'''
On this Page there can be ask Query Situation. 
Query: Waht the Propablity is for the Attributes to be in the Range/Style 
Evidence: What is Given for the Probablity of the Query 
After = Button there will be the Display of the Percantage 
'''


global modal_var_index
modal_var_index = -1

global modal_type
# 0 = q and 1 = e
modal_type = -1

global modal_basic_que
modal_basic_que = c.gen_modal_basic_id("_que")

modal_option_que = c.gen_modal_option_id("_que")

dash.register_page(__name__)

def layout_que():
    '''
        Generates the Basic Layout in Dash for Query withe the Tree varnames as Options
    :return: Dash html strucktur
    '''
    return dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H1("Query", className='text-center mb-4'), width=12),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([
                    html.Div("P ", className="align-self-center text-end float-end",
                             style={'fontSize': 40, 'padding-top': 0}),
                ], id="text_l_que", align="center", className="", width=2),
                dbc.Col(id="q_variable_que",
                        children=[dcc.Dropdown(id={'type': 'dd_q_que', 'index': 0}, options=sorted(c.in_use_tree.varnames))],
                        width=1, className="d-grid gap-3 border-start border-secondary border-3 rounded-4"),
                dbc.Col(id="q_input_que",
                        children=[dcc.Dropdown(id={'type': 'i_q_que', 'index': 0}, disabled=True)], width=3,
                        className="d-grid gap-3 "),
                dbc.Col(children=[html.Div(id="q_option_que", children=[
                    dbc.Button("👁️", id=dict(type='b_q_que', index=0), disabled=True, n_clicks=0, className="me-2 mb-3",
                               size="sm")], className=" d-grid align-self-start")
                                  ],
                        className="d-grid gap-0 gx-0 d-flex align-items-stretch flex-grow-0 align-self-stretch border-end border-3 border-secondary "),
                dbc.Col(id="e_variable_que",
                        children=[dcc.Dropdown(id={'type': 'dd_e_que', 'index': 0}, options=sorted(c.in_use_tree.varnames))],
                        width=1, className="d-grid gap-0 border-start border-3 border-secondary ps-3"),
                dbc.Col(id="e_input_que",
                        children=[dcc.Dropdown(id={'type': 'i_e_que', 'index': 0}, disabled=True)], width=3,
                        className="d-grid gap-3 "),
                dbc.Col(children=[html.Div(id="e_option_que", children=[
                    dbc.Button("👁️", id=dict(type='b_e_que', index=0), disabled=True, n_clicks=0, className="me-2 mb-3",
                               size="sm")],
                                           className=" d-grid border-end border-secondary border-3 rounded-4")
                                  ],
                        className="d-grid gx-1 d-md-flex align-self-center"),
            ], className="justify-content-center",
        ),
        dbc.Row(dbc.Button("=", id="erg_b_que", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
        dbc.Row(dbc.Col(html.Div("", id="erg_text_que", className="fs-1 text text-center pt-3 "))),
        modal_option_que
    ], fluid=True
)

layout = layout_que

def query_gen(dd_vals: List, q_var: List, q_in: List, q_op):
    '''
        Handel all action in the Query Part of the GUI (Extend Change Reduce)
    :param dd_vals: All Varietals used in Query Section are chosen
    :param q_var: the Dropdown of variable of Query Section
    :param q_in: the Input for the Variables of Query Section
    :param q_op:  the Variabel  who is selected for the Zoom
    :return: Updatet Varibel List and the Input.
    '''
    cb = ctx.triggered_id
    if dd_vals[cb.get("index")] is None:
        return c.del_selector_from_div_button(c.in_use_tree, q_var, q_in, q_op, cb.get("index"))

    variable = c.in_use_tree.varnames[dd_vals[cb.get("index")]]


    if variable.numeric:
        minimum = c.priors[variable.name].cdf.intervals[0].upper
        maximum = c.priors[variable.name].cdf.intervals[-1].lower

        q_in[cb.get("index")] = c.create_range_slider(minimum, maximum, id={'type': 'i_q_que', 'index': cb.get("index")},
                                                      tooltip={"placement": "bottom", "always_visible": False})

    elif variable.symbolic:
        q_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_q_que", "index": cb.get("index")},
                                             options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                           variable.domain.labels.values())},
                                             value=list(variable.domain.labels.values()),
                                             multi=True, )  # list(variable.domain.labels.keys())
    elif variable.integer:
        lab = list(variable.domain.labels.values())
        mini = min(lab)
        maxi = max(lab)
        markings = dict(zip(lab, map(str, lab)))
        q_in[cb.get("index")] = c.create_range_slider(minimum=mini-1, maximum=maxi+1, value=[mini, maxi],
                                                      id={'type': 'i_q_que', 'index': cb.get("index")}, dots=False, marks = markings,
                                                      tooltip={"placement": "bottom", "always_visible": False})

    if len(q_var) - 1 == cb.get("index"):
        return c.add_selector_to_div_button(c.in_use_tree, q_var, q_in, q_op, 'q_que', cb.get("index") + 1)
    return c.update_free_vars_in_div(c.in_use_tree, q_var), q_in, q_op


def evid_gen(dd_vals, e_var, e_in, e_op):
    '''
        Handel all action in the Evidence Part of the GUI (Extend Change Reduce)
    :param dd_vals: All Varietals used in Evidence Section are chosen
    :param e_var: the Dropdown of variable of Evidence Section
    :param e_in: the Input for the Variables of Evidence Section
    :param q_op:  the Variabel  who is selected for the Zoom
    :return: Updatet Varibel List and the Input.
    '''
    e_var: List[dict] = e_var
    e_in: List[dict] = e_in
    cb = ctx.triggered_id
    print(cb)
    if dd_vals[cb.get("index")] is None:
        return c.del_selector_from_div_button(c.in_use_tree, e_var, e_in, e_op, cb.get('index'))

    variable = c.in_use_tree.varnames[dd_vals[cb.get("index")]]
    if variable.numeric:
        minimum = c.priors[variable.name].cdf.intervals[0].upper
        maximum = c.priors[variable.name].cdf.intervals[-1].lower
        e_in[cb.get("index")] = c.create_range_slider(minimum, maximum, id={'type': 'i_e_que', 'index': cb.get("index")},
                                                      tooltip={"placement": "bottom", "always_visible": False})
    elif variable.symbolic:
        e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e_que", "index": cb.get("index")},
                                             options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                           variable.domain.labels.values())},
                                             value=list(variable.domain.labels.values()), multi=True, )
    elif variable.integer:
        lab = list(variable.domain.labels.values())
        mini = min(lab)
        maxi = max(lab)
        markings = dict(zip(lab, map(str, lab)))
        e_in[cb.get("index")] = c.create_range_slider(minimum=mini - 1, maximum=maxi + 1, value=[mini, maxi],
                                                      id={'type': 'i_e_que', 'index': cb.get("index")}, dots=False,
                                                      marks=markings,
                                                      tooltip={"placement": "bottom", "always_visible": False})
    if len(e_var) - 1 == cb.get("index"):
        return c.add_selector_to_div_button(c.in_use_tree, e_var, e_in, e_op, "e_que", cb.get("index") + 1)
    return c.update_free_vars_in_div(c.in_use_tree, e_var), e_in, e_op


@callback(
    Output('q_variable_que', 'children'),
    Output('q_input_que', 'children'),
    Output('q_option_que', 'children'),
    Output('e_variable_que', 'children'),
    Output('e_input_que', 'children'),
    Output('e_option_que', 'children'),
    Output('text_l_que', 'children'),
    Output('modal_option_que', 'children'),
    Output('modal_option_que', 'is_open'),
    Input({'type': 'dd_q_que', 'index': ALL}, 'value'),
    Input({'type': 'dd_e_que', 'index': ALL}, 'value'),
    Input({'type': 'b_q_que', 'index': ALL}, 'n_clicks'),
    Input({'type': 'b_e_que', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_save_que', 'index': ALL}, 'n_clicks'),
    State('q_variable_que', 'children'),
    State('q_input_que', 'children'),
    State('e_variable_que', 'children'),
    State('e_input_que', 'children'),
    State('q_option_que', 'children'),
    State('e_option_que', 'children'),
    State({'type': 'op_i_que', 'index': ALL}, 'value'),
)
def query_router(q_dd, e_dd, b_q, b_e, op_s, q_var, q_in, e_var, e_in, q_op, e_op, op_i):
    '''
        Receives app callback events and manages/redirects these to the correct functions.
    :param q_dd: Query Varibels Names
    :param e_dd: Evidence Variable Names
    :param b_q: Trigger if the Zoom Button in the Query is Pressed
    :param b_e: Trigger if the Zoom Button in the Evidence is Pressed
    :param op_s: Trigger if the Modal parameter from a Zoom should be saved
    :param q_var: Div of the Query Variable
    :param q_in: Div or the Input of Query
    :param e_var: Div of the Evidence Variable
    :param e_in: Div or the Input of Evidence
    :param q_op: Information of whiche Zoom Button was pressed in the Query section
    :param e_op: Information of whiche Zoom Button was pressed in the Evidence section
    :param op_i: The Values choosen in the Zoom Modal
    :return: Query Varibels, Query Input, Evidence Variable, Evidence Input, Text Prefix.
    '''
    global modal_var_index
    global modal_type  # 0 = q and 1 = e

    cb = ctx.triggered_id if not None else None
    if cb is None:
        return q_var, q_in, q_op, e_var, e_in, e_op, \
            c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False

    elif cb.get("type") == "dd_q_que":
        return *query_gen(q_dd, q_var, q_in, q_op), e_var, e_in, e_op, \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False
    elif cb.get("type") == "dd_e_que":

        return q_var, q_in, q_op, *evid_gen(e_dd, e_var, e_in, e_op), \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False
    elif cb.get("type") == "b_e_que" and e_dd[cb.get("index")] != []:

        modal_var_index = cb.get("index")
        modal_type = 1
        #Wenn kein Value exestiert wird min max genommen
        modal_body = c.generate_modal_option(model=c.in_use_tree, var=e_var[cb.get("index")]['props']['value'],
                                             value=e_in[cb.get("index")]['props']
                                             .get('value', [e_in[cb.get("index")]['props']['min'],
                                                           e_in[cb.get("index")]['props']['max']]),
                                             priors=c.priors, id="_que")
        return q_var, q_in, q_op, e_var, e_in, e_op, \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_body, True
    elif cb.get("type") == "b_q_que" and q_dd[cb.get("index")] != []:

        modal_var_index = cb.get("index")
        modal_type = 0


        modal_body = c.generate_modal_option(model=c.in_use_tree, var=q_var[cb.get("index")]['props']['value'],
                                             value=q_in[cb.get("index")]['props'].get('value',
                                                                                      [q_in[cb.get("index")]['props']['min'], q_in[cb.get("index")]['props']['max']]), priors=c.priors, id="_que")
        return q_var, q_in, q_op, e_var, e_in, e_op, \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_body, True
    elif cb.get("type") == "option_save_que":
        new_vals = c.fuse_overlapping_range(op_i)
        if modal_type == 1:
            e_in[modal_var_index]['props']['value'] = new_vals
            e_in[modal_var_index]['props']['drag_value'] = new_vals
            return q_var, q_in, q_op, e_var, e_in, e_op, \
                   c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False
        else:
            q_in[modal_var_index]['props']['value'] = new_vals
            q_in[modal_var_index]['props']['drag_value'] = new_vals
            return q_var, q_in, q_op, e_var, e_in, e_op, \
                   c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False
    else:
        return q_var, q_in, q_op, e_var, e_in, e_op, \
               c.create_prefix_text_query(len_fac_q=len(q_var), len_fac_e=len(e_var)), modal_basic_que, False


@callback(
    Output("modal_input_que", "children"),
    Input("op_add_que", "n_clicks"),
    Input({'type': 'op_i_que', 'index': ALL}, 'value'),
    State("modal_input_que", "children"),
    State({'type': 'dd_e_que', 'index': ALL}, 'value'),
    State({'type': 'dd_q_que', 'index': ALL}, 'value'),
)
def modal_router(op, op_i, m_bod, dd_e, dd_q):
    '''
        Recessive all App Calls that are change the Modal for the zoom Function
    :param op: Trigger to add More Input Option by Numeric Variabel
    :param op_i: Trigger to update Chance for the Chosen values
    :param m_bod: The State of the Modal
    :param dd_e: div withe the chosen values in the Evidence Section
    :param dd_q: div withe the chosen values in the Query Section
    :return: update Modal Body for the Zoom
    '''
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return m_bod
    global modal_var_index
    global modal_type
    dd = dd_e if modal_type == 1 else dd_q
    var = dd[modal_var_index]
    if not isinstance(m_bod, list):
        m_in_new = [m_bod]
    else:
        m_in_new = m_bod
    if cb == "op_add_que":
        index = m_in_new[-2]['props']['children'][0]['props']['children'][1]['props']['id']['index']
        type = m_in_new[1]['props']['children'][0]['props']['children'][1]['type']
        if type == "RangeSlider":

            mini = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['min']
            maxi = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['max']
            range_string = html.Div(f"Range {index + 2}",
                                    style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal)-1)]))
            n_slider = c.create_range_slider(minimum=mini, maximum=maxi,id={'type': 'op_i_que', 'index': index + 1},
                                             value=[mini, maxi], dots=False,
                                             tooltip={"placement": "bottom", "always_visible": False},
                                             className="flex-fill")
            var_map = c.div_to_variablemap(c.in_use_tree, [var], [[mini, maxi]])
            prob = c.in_use_tree.infer(var_map, {})

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
        var_map = c.div_to_variablemap(c.in_use_tree, [var], [value])
        prob = c.in_use_tree.infer(var_map, {})
        prob_div = html.Div(f"{prob}", style=dict(color=c.color_list_modal[id % (len(c.color_list_modal)-1)]))
        m_in_new[id + 1]['props']['children'][0]['props']['children'][2] = prob_div
        return m_in_new


@callback(
    Output("erg_text_que", "children"),
    Input("erg_b_que", "n_clicks"),

    State({'type': 'dd_q_que', 'index': ALL}, 'value'),
    State({'type': 'i_q_que', 'index': ALL}, 'value'),
    State({'type': 'dd_e_que', 'index': ALL}, 'value'),
    State({'type': 'i_e_que', 'index': ALL}, 'value'),
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
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return ""
    query = c.div_to_variablemap(c.in_use_tree, q_var, q_in)
    evidence = c.div_to_variablemap(c.in_use_tree, e_var, e_in)
    print(f'query:{query}, evi:{evidence}')
    try:
        result = c.in_use_tree.infer(query, evidence)

    except Exception as e:
        print(e)
        return "Unsatasfiable"
    print(result)
    return "{}%".format(round(result * 100, 2))


# 2. Posterior RESULTS
# 3. MASKE DTAILS GUI
# 4. MEHER SLIDER ODER IN MPE UND QUERY
# 5. https://observablehq.com/@d3/tree-of-life
# PIP
# JPT Update
# Landing page
