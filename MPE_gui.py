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
import json
from typing import List


global model
model: jpt.trees.JPT = c.default_tree

global priors
priors = model.independent_marginals()

global result

global page
page = 0

global modal_var_index
modal_var_index = -1


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
                dbc.Col(html.H1("Most Probable Explanation", className='text-center mb-4'), width=12),
                dbc.Col(dcc.Upload(children=dbc.Button("🌱", n_clicks=0, className="position-absolute top-0 end-0"),
                                   id="upload_tree"))
            ]
        ),
        dbc.Row(
            [
                dbc.Col([
                    html.Div("argmax ", className="pe-3",
                             style={'fontSize': 20, 'padding-top': 0}),
                    html.Div("P ", className="ps-3",
                             style={'fontSize': 30, 'padding-top': 0}),
                ], id="text_l", align="center",
                    className="d-flex flex-wrap align-items-center justify-content-end pe-3", width=2),
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
                        className="d-grid gap-3 "),
                dbc.Col(children=[html.Div(id="e_option", children=[
                    dbc.Button("👁️", id=dict(type='b_e', index=0), disabled=True, n_clicks=0, className="me-2 mb-3",
                               size="sm")], className=" d-grid border-end border-secondary border-3 rounded-4")
                                  ],
                        width=1, className="d-grid gx-1 d-md-flex align-self-center"),
            ], className="justify-content-md-center"
        ),
        dbc.Row(dbc.Button("=", id="erg_b", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)),
        dbc.Row(
            [
                dbc.Col(dbc.Button("<", id="b_erg_pre", n_clicks=0, disabled=True),
                        className="d-flex justify-content-end align-self-stretch"),
                dbc.Col(children=[], id="mpe_erg", className=""),
                dbc.Col(dbc.Button(">", id="b_erg_next", n_clicks=0, disabled=True),
                        className="d-flex justify-content-start align-self-stretch")
            ], className="pt-3", align="center"),
        dbc.Row(),
        modal_option
    ], fluid=True
)


@app.callback(
    Output('e_variable', 'children'),
    Output('e_input', 'children'),
    Output('e_option', 'children'),
    Output('text_l', 'children'),
    Output('q_variable', 'children'),
    Output('modal_option', 'children'),
    Output('modal_option', 'is_open'),
    Input("upload_tree", 'contents'),
    Input({'type': 'dd_e', 'index': ALL}, 'value'),
    Input({'type': 'b_e', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_save', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_abort', 'index': ALL}, 'n_clicks'),
    State('e_variable', 'children'),
    State('e_input', 'children'),
    State('q_variable', 'children'),
    State('e_option', 'children'),
    State({'type': 'op_i', 'index': ALL}, 'value'),
)
def evid_gen(upload, dd_vals, b_e, op_s, op_a, e_var, e_in, q_var, e_op, op_i):
    """
    Receives appCallback events and manages these to the correct
    :param upload: Path to the new jpt Tree as a File
    :param dd_vals: All Varietals used in Evidence Section are chosen
    :param e_var: the Dropdown of variable of Evidence Section
    :param e_in: the Input for the Variables of Evidence Section
    :param q_var: the Dropdown of variable of Query Section
    :return: Updatet Varibel List and the Input.
    """

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
            return e_var, e_in, e_op, c.create_prefix_text_mpe(len(e_var)), q_var, modal_basic, False
        model = io_model
        priors = model.independent_marginals()
        q_var_n = dcc.Dropdown(id="text_var", options=sorted(model.varnames), value=sorted(model.varnames),
                               multi=True, disabled=True)
        return *c.reset_gui_button(io_model, "e"), c.create_prefix_text_mpe(1), q_var_n, modal_basic, False
    elif cb.get("type") == "dd_e":
        if dd_vals[cb.get("index")] is None:
            return *c.del_selector_from_div_button(model, e_var, e_in, e_op, cb.get("index")), \
                   c.create_prefix_text_mpe(4), q_var, modal_basic, False

        variable = model.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
            minimum = priors[variable].cdf.intervals[0].upper
            maximum = priors[variable].cdf.intervals[-1].lower
            e_in[cb.get("index")] = c.create_range_slider(minimum, maximum, id={'type': 'i_e', 'index': cb.get("index")}
                                                          , dots=False,
                                                          tooltip={"placement": "bottom", "always_visible": False})

        elif variable.symbolic:
            e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e", "index": cb.get("index")},
                                                 options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                               variable.domain.labels.values())},
                                                 value=list(variable.domain.labels.values()),
                                                 multi=True, )

        if len(e_var) - 1 == cb.get("index"):
            return *c.add_selector_to_div_button(model, e_var, e_in, e_op, "e", cb.get("index") + 1), \
                   c.create_prefix_text_mpe(len(e_var)), q_var, modal_basic, False
    elif cb.get("type") == "b_e" and dd_vals[cb.get("index")] != []:
        # Dont Like dont know to do it other wise
        global modal_var_index
        modal_var_index = cb.get("index")
        modal_body = c.generate_modal_option(model=model, var=e_var[cb.get("index")]['props']['value'],
                                           value=e_in[cb.get("index")]['props']['drag_value'], priors=priors)
        return e_var, e_in, e_op, c.create_prefix_text_mpe(len(e_var)), q_var, modal_body, True
    elif cb.get("type") == "option_save":
        new_vals = c.fuse_overlapping_range(op_i)
        e_in[modal_var_index]['props']['value'] = new_vals
        e_in[modal_var_index]['props']['drag_value'] = new_vals
        return e_var, e_in, e_op, c.create_prefix_text_mpe(len(e_var)), q_var, modal_basic, False

    return c.update_free_vars_in_div(model, e_var), e_in, e_op, c.create_prefix_text_mpe(len(e_var)), \
           q_var, modal_basic, False

@app.callback(
    Output("mod_in", "children"),
    Input("op_add", "n_clicks"),
    State("mod_in", "children"),
)
def modal_router(op, m_in):
    if not isinstance(m_in, list):
        m_in_new = [m_in]
    else:
        m_in_new = m_in
    index = m_in_new[0]['props']['id']['index']
    type = m_in_new[0]['type']
    if type == "RangeSlider":
        min = m_in_new[0]['props']['min']
        max = m_in_new[0]['props']['max']
        m_in_new.append(c.create_range_slider(minimum=min, maximum=max, id=dict(type="op_i", index=index+1),
                                              value=[min, max], dots=False,
                                              tooltip={"placement": "bottom", "always_visible": False}))
    return m_in_new



@app.callback(
    Output('mpe_erg', 'children'),
    Output('b_erg_pre', 'disabled'),
    Output('b_erg_next', 'disabled'),
    Input('erg_b', 'n_clicks'),
    Input('b_erg_pre', 'n_clicks'),
    Input('b_erg_next', 'n_clicks'),
    State({'type': 'dd_e', 'index': ALL}, 'value'),
    State({'type': 'i_e', 'index': ALL}, 'value'),
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
        if len(result) > page + 1:
            return mpe(result[page]), False, False
        else:
            return mpe(result[page]), False, True
    else:
        page = 0
        evidence_dict = c.div_to_variablemap(model, e_var, e_in)
        try:
            result = model.mpe(evidence=jpt.variables.VariableMap(evidence_dict.items()))
            print(result[0].maximum)

        except Exception as e:
            print("Error was", type(e), e)
            return [html.Div("Unsatisfiable", className="fs-1 text text-center pt-3 ")], True, True
        if len(result) > 1:
            return mpe(result[0]), True, False
        else:
            return mpe(result[0]), True, True


def mpe(res):
    """
    Generates the Result from Res of a Variable
    :param res:  Results of a specific Variable
    :return: Div around the generated mpe Result of the Variable
    """
    return c.mpe_result_to_div(model, res)


if __name__ == '__main__':
    app.run_server(debug=True)
