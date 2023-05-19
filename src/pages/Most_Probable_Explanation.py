from typing import List

import jpt
import jpt.variables
import dash_bootstrap_components as dbc
import dash
from dash import dcc, html, Input, Output, State, ctx, ALL, callback
import components as c

"""
    Most Probable Explanation GUI here can be chosen which Variabel to be consider (Default are all)
    Left kan chosen what be Given Information in the Moment. 
    After the Equals Button the Values will be Displayed, that can be change if Multi Results are exsisting. 
"""

global maxima

global page
page = 0

global likelihood
likelihood = 0.0

global modal_var_index
modal_var_index = -1

global modal_basic_mpe
modal_basic_mpe = c.gen_modal_basic_id("_mpe")

modal_option_mpe = c.gen_modal_option_id("_mpe")

dash.register_page(__name__)


def layout_mpe():
    """
        Generad the Default style for the MPE Gui
    :return:  Dash HTML Construkt
    """
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.H1("Most Probable Explanation", className='text-center mb-4'), width=12),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col([
                        html.Div("argmax ", className="pe-3",
                                 style={'fontSize': 20, 'padding-top': 0}),
                        html.Div("P ", className="ps-3",
                                 style={'fontSize': 30, 'padding-top': 0}),
                    ], id="text_l_mpe", align="center",
                        className="d-flex flex-wrap align-items-center justify-content-end pe-3", width=2), #d-flex flex-wrap align-items-center justify-content-end pe-3
                    dbc.Col(id="q_variable_mpe",
                            children=[
                                dcc.Dropdown(id="text_var_mpe", options=sorted(c.in_use_tree.varnames),
                                             value=sorted(c.in_use_tree.varnames),
                                             multi=True, disabled=True)],
                            width=4, className="row row-cols-1 g-1 gy-2 align-items-center border-start border-3 rounded-4 border-secondary"), #d-grid gap-3
                    dbc.Col(id="e_variable_mpe",
                            children=[dcc.Dropdown(id={'type': 'dd_e_mpe', 'index': 0},
                                                   options=sorted(c.in_use_tree.varnames))],
                            width=2, className="row row-cols-1 g-1 gy-2 align-items-center border-start border-3 border-secondary"), #d-grid gap-3 ps-3
                    dbc.Col(id="e_input_mpe",
                            children=[dcc.Dropdown(id={'type': 'i_e_mpe', 'index': 0}, disabled=True)], width=3,
                            className="row row-cols-1 g-1 gy-2 align-items-center"),
                    dbc.Col(id="e_option_mpe", children=[
                        dbc.Button("👁️", id=dict(type='b_e_mpe', index=0), disabled=True, n_clicks=0,
                                   className="",
                                   size="sm")], width=1, className="row row-cols-1 g-1 gy-2 align-items-center pe-3 ps-1 border-end border-secondary border-3 rounded-4"), #d-grid border-end border-secondary border-3 rounded-4#d-grid gx-1 d-md-flex align-self-center
                ], className="row row-cols-6 g-1 gy-2 mb-3" #justify-content-md-center
            ),
            dbc.Row(dbc.Button("=", id="erg_b_mpe", className="d-grid gap-2 col-6 mx-auto", n_clicks=0)), #d-grid gap-2 col-3 mt-3 mx-auto
            dbc.Row(
                [
                    dbc.Col(dbc.Button("<", id="b_erg_pre_mpe", n_clicks=0, disabled=True),
                            className="d-flex justify-content-end align-self-stretch"),
                    dbc.Col(children=[], id="mpe_erg", className=""),
                    dbc.Col(dbc.Button(">", id="b_erg_next_mpe", n_clicks=0, disabled=True),
                            className="d-flex justify-content-start align-self-stretch")
                ], className="pt-3", align="center"),
            dbc.Row(),
            modal_option_mpe
        ], fluid=True
    )


layout = layout_mpe


@callback(
    Output('e_variable_mpe', 'children'),
    Output('e_input_mpe', 'children'),
    Output('e_option_mpe', 'children'),
    Output('text_l_mpe', 'children'),
    Output('q_variable_mpe', 'children'),
    Output('modal_option_mpe', 'children'),
    Output('modal_option_mpe', 'is_open'),
    Input({'type': 'dd_e_mpe', 'index': ALL}, 'value'),
    Input({'type': 'b_e_mpe', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_save_mpe', 'index': ALL}, 'n_clicks'),
    State('e_variable_mpe', 'children'),
    State('e_input_mpe', 'children'),
    State('q_variable_mpe', 'children'),
    State('e_option_mpe', 'children'),
    State({'type': 'op_i_mpe', 'index': ALL}, 'value'),
)
def evid_gen(dd_vals, b_e, op_s, e_var, e_in, q_var, e_op, op_i):
    """
        Receives appCallback events and manages these to the correct
    :param dd_vals: All Varietals used in Evidence Section are chosen
    :param b_e: Trigger if the Zoom Button in the Evidence is Pressed
    :param op_s: Trigger if the Modal parameter from a Zoom should be saved
    :param e_var: the Dropdown of variable of Evidence Section
    :param e_in: the Input for the Variables of Evidence Section
    :param q_var: the Dropdown of variable of Query Section
    :param e_op: Information of whiche Zoom Button was pressed in the Evidence section
    :param op_i: The Values choosen in the Zoom Modal
    :return: Updatet Varibel List and the Input.
    """

    cb = ctx.triggered_id if not None else None
    if cb is None:
        return e_var, e_in, e_op, c.create_prefix_text_mpe(len(e_var)), q_var, modal_basic_mpe, False
    elif cb.get("type") == "dd_e_mpe":
        if dd_vals[cb.get("index")] is None:
            return *c.del_selector_from_div_button(c.in_use_tree, e_var, e_in, e_op, cb.get("index")), \
                c.create_prefix_text_mpe(4), q_var, modal_basic_mpe, False

        variable = c.in_use_tree.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
            minimum = c.priors[variable.name].cdf.intervals[0].upper
            maximum = c.priors[variable.name].cdf.intervals[-1].lower
            e_in[cb.get("index")] = c.create_range_slider(minimum, maximum,
                                                          id={'type': 'i_e_mpe', 'index': cb.get("index")}
                                                          , dots=False,
                                                          tooltip={"placement": "bottom", "always_visible": False})

        elif variable.symbolic:
            e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e_mpe", "index": cb.get("index")},
                                                 options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                               variable.domain.labels.values())},
                                                 value=list(variable.domain.labels.values()),
                                                 multi=True, )
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
            return *c.add_selector_to_div_button(c.in_use_tree, e_var, e_in, e_op, "e_mpe", cb.get("index") + 1), \
                c.create_prefix_text_mpe(len(e_var)), q_var, modal_basic_mpe, False
    elif cb.get("type") == "b_e_mpe" and dd_vals[cb.get("index")] != []:
        # Dont Like dont know to do it other wise
        global modal_var_index
        modal_var_index = cb.get("index")
        variable = c.in_use_tree.varnames[dd_vals[cb.get("index")]]
        modal_body = List
        if variable.numeric:
            modal_body = c.generate_modal_option(model=c.in_use_tree, var=e_var[cb.get("index")]['props']['value'],
                                                 value=[e_in[cb.get("index")]['props']['min'],
                                                        e_in[cb.get("index")]['props']['max']],
                                                 priors=c.priors, id="_mpe")
        elif variable.symbolic or variable.integer:
            modal_body = c.generate_modal_option(model=c.in_use_tree, var=e_var[cb.get("index")]['props']['value'],
                                                 value=e_in[cb.get("index")]['props'].get('value'), priors=c.priors,
                                                 id="_mpe")

        return e_var, e_in, e_op, c.create_prefix_text_mpe(len(e_var)), q_var, modal_body, True
    elif cb.get("type") == "option_save_mpe":
        new_vals = List
        variable = c.in_use_tree.varnames[dd_vals[cb.get("index")]]
        if variable.numeric or variable.integer:
            new_vals = c.fuse_overlapping_range(op_i)
        else:
            new_vals = op_i[0]  # is List of a List
        e_in[modal_var_index]['props']['value'] = new_vals
        return e_var, e_in, e_op, c.create_prefix_text_mpe(len(e_var)), q_var, modal_basic_mpe, False

    return c.update_free_vars_in_div(c.in_use_tree, e_var), e_in, e_op, c.create_prefix_text_mpe(len(e_var)), \
        q_var, modal_basic_mpe, False


@callback(
    Output("modal_input_mpe", "children"),
    Input("op_add_mpe", "n_clicks"),
    Input({'type': 'op_i_mpe', 'index': ALL}, 'value'),
    State("modal_input_mpe", "children"),
    State({'type': 'dd_e_mpe', 'index': ALL}, 'value'),
)
def modal_router(op, op_i, m_bod, dd):
    """
        Recessive all App Calls that are change the Modal for the zoom Function
    :param op: Trigger to add More Input Option by Numeric Variabel
    :param op_i: Trigger to update Chance for the Chosen values
    :param m_bod: The State of the Modal
    :param dd: div withe the chosen values
    :return: update Modal Body for the Zoom
    """
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return m_bod
    global modal_var_index
    var = dd[modal_var_index]
    variable = c.in_use_tree.varnames[var]
    if not isinstance(m_bod, list):
        m_in_new = [m_bod]
    else:
        m_in_new = m_bod
    if cb == "op_add_mpe":
        index = m_in_new[-2]['props']['children'][0]['props']['children'][1]['props']['id']['index']
        type = m_in_new[1]['props']['children'][0]['props']['children'][1]['type']
        if variable.numeric:

            mini = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['min']
            maxi = m_in_new[1]['props']['children'][0]['props']['children'][1]['props']['max']
            range_string = html.Div(f"Range {index + 2}",
                                    style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal) - 1)]))
            n_slider = c.create_range_slider(minimum=mini, maximum=maxi, id={'type': 'op_i_mpe', 'index': index + 1},
                                             value=[mini, maxi], dots=False,
                                             tooltip={"placement": "bottom", "always_visible": False},
                                             className="flex-fill")
            var_map = c.div_to_variablemap(c.in_use_tree, [var], [[mini, maxi]])
            prob = c.in_use_tree.infer(var_map, {})

            prob_div = html.Div(f"{prob}",
                                style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal) - 1)]))
            m_in_new.insert(len(m_in_new) - 1, dbc.Row([
                html.Div([range_string, n_slider, prob_div],
                         id=f"modal_color_{(index + 1) % (len(c.color_list_modal) - 1)}",
                         className="d-flex flex-nowrap justify-content-center ps-2")
            ], className="d-flex justify-content-center"))
            return m_in_new
        elif variable.integer:
            lab = list(variable.domain.labels.values())
            mini = min(lab)
            maxi = max(lab)
            markings = dict(zip(lab, map(str, lab)))
            range_string = html.Div(f"Range {index + 2}",
                                    style=dict(color=c.color_list_modal[(index + 1) % (len(c.color_list_modal)-1)]))
            n_slider = c.create_range_slider(minimum=mini, maximum=maxi, value=[mini, maxi]
                                                          ,id={'type': 'op_i_mpe', 'index': index + 1}, dots=False,
                                                          marks=markings,
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
        prob_div = html.Div(f"{prob}", style=dict(color=c.color_list_modal[id % (len(c.color_list_modal) - 1)]))
        m_in_new[id + 1]['props']['children'][0]['props']['children'][2] = prob_div
        return m_in_new


@callback(
    Output('mpe_erg', 'children'),
    Output('b_erg_pre_mpe', 'disabled'),
    Output('b_erg_next_mpe', 'disabled'),
    Input('erg_b_mpe', 'n_clicks'),
    Input('b_erg_pre_mpe', 'n_clicks'),
    Input('b_erg_next_mpe', 'n_clicks'),
    State({'type': 'dd_e_mpe', 'index': ALL}, 'value'),
    State({'type': 'i_e_mpe', 'index': ALL}, 'value'),
)
def erg_controller(n1, n2, n3, e_var, e_in):
    """
        Manages the MPE Reulst and the Switch if possible between Results
    :param n1: event for generating Result
    :param n2: the Previous Result
    :param n3: the Next Result
    :param e_var: the Dropdown of variable of Evidence Section
    :param e_in: the Input for the Variables of Evidence Section
    :return: Div of the Result and if Previous or Next Result exists
    """
    global maxima
    global page
    global likelihood
    cb = ctx.triggered_id if not None else None
    if cb is None:
        return [], True, True
    if cb == "b_erg_pre_mpe":
        page -= 1
        if page == 0:
            return mpe(maxima[page], likelihood), True, False
        else:
            return mpe(maxima[page], likelihood), False, False
    elif cb == "b_erg_next_mpe":
        page += 1
        if len(maxima) > page + 1:
            return mpe(maxima[page], likelihood), False, False
        else:
            return mpe(maxima[page], likelihood), False, True
    else:
        page = 0
        evidence_dict = c.div_to_variablemap(c.in_use_tree, e_var, e_in)
        try:
            evi = jpt.variables.LabelAssignment(evidence_dict.items())
            maxima, likelihood = c.in_use_tree.mpe(evidence=evi)

        except Exception as e:
            print("Error was", type(e), e)
            return [html.Div("Unsatisfiable", className="fs-1 text text-center pt-3 ")], True, True
        if len(maxima) > 1:
            return mpe(maxima[0], likelihood), True, False
        else:
            return mpe(maxima[0], likelihood), True, True


def mpe(res, likelihood):
    """
        Generates the Result from Res of a Variable
    :param res:  Results of a specific Variable
    :param likelihood: The likelihood of the maxima
    :return: Div around the generated mpe Result of the Variable
    """
    return c.mpe_result_to_div(c.in_use_tree, res, likelihood)
