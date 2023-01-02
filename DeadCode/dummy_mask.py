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
from typing import List

global model
import json

model: jpt.trees.JPT = c.default_tree

global priors
priors = model.independent_marginals()

global result

global page
page = 0

global modal_var_index
modal_var_index = -1

#TEMP
global modal_basic
modal_basic = [
        dbc.ModalHeader(dbc.ModalTitle('temp')),
        dbc.ModalBody([
            html.Div([dcc.Dropdown(id={'type': 'op_i', 'index': 0}), dbc.Button(id="op_add")], id="mod_in")
        ]),
        dbc.ModalFooter(
            [
                dbc.Button("Abort", id=dict(type="option_abort", index=0), className="ms-auto", n_clicks=0),
                dbc.Button("Save", id=dict(type="option_save", index=0), className="ms-auto", n_clicks=0)
            ]
        ),
    ]



app = dash.Dash(__name__, prevent_initial_callbacks=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )

modal_option = dbc.Modal(
    [
        # #Chidlren? alles Generieren
        dbc.ModalHeader(dbc.ModalTitle('temp')),
        dbc.ModalBody([
            html.Div([dcc.Dropdown(id={'type': 'op_i', 'index': 0}), dbc.Button(id="op_add")], id="mod_in")
        ]),
        dbc.ModalFooter(
            [
                dbc.Button("Abort", id=dict(type="option_abort", index=0), className="ms-auto", n_clicks=0),
                dbc.Button("Save", id=dict(type="option_save", index=0), className="ms-auto", n_clicks=0)
            ]
        ),
    ],
    id="modal_option", is_open=False, size="xl", backdrop="static"
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
                        className="d-grid gx-0 ps-2 "),
                dbc.Col(children=[html.Div(id="e_option", children=[
                    dbc.Button("👁️", id=dict(type='b_e', index=0), disabled=True, n_clicks=0, className="me-2 mb-3",
                               size="sm")], className=" d-grid border-end border-secondary border-3 rounded-4")
                                  ],
                        width=1, className="d-grid gx-1 d-md-flex align-self-center"),
            ], className="justify-content-md-center"
        ),
        modal_option,
    ], fluid=True,
)


def add_selector_to_div_button(model: jpt.trees.JPT, variable_div, constrains_div, option_div, type: str,
                               index: int) \
        -> (List[dcc.Dropdown], List, List):
    """
    :param model: the JPT model of the Prob. Tree
    :param variable_div: list of Components to Chose Variable in the GUI
    :param constrains_div: list of Components that are the Constraints for the Variables on the Same Index
    :param type: the Type of the Component for the ID
    :param index: the index Number of the Component for the ID
    :return: Variable Children and Constrains Children for the GUI withe one more Row
    """
    variable_list = variable_div
    constrains_list = constrains_div
    option_list = option_div

    variable_list = c.update_free_vars_in_div(model, variable_list)
    option_list[-1]['props']['disabled'] = False

    variable_list.append(
        dcc.Dropdown(id={'type': f'dd_{type}', 'index': index},
                     options=variable_list[0]['props']['options'][1:]))
    constrains_list.append(dcc.Dropdown(id={'type': f'i_{type}', 'index': index}, disabled=True))
    option_list.append(
        dbc.Button("👁️", id=dict(type=f'b_{type}', index=index), disabled=True, n_clicks=0, className="me-2 mb-3",
                   size="sm"))
    return variable_list, constrains_list, option_list


def reset_gui_button(model: jpt.trees.JPT, type: str):
    var_div = [dcc.Dropdown(id={'type': f'dd_{type}', 'index': 0}, options=sorted(model.varnames))]
    in_div = [dcc.Dropdown(id={'type': f'i_{type}', 'index': 0}, disabled=True)]
    op_div = [dbc.Button("👁️", id=dict(type='b_e', index=0), disabled=True, n_clicks=0, className="me-2",
                         size="sm")]
    return var_div, in_div, op_div


def del_selector_from_div_button(model: jpt.trees.JPT, variable_div: List, constrains_div: List, option_div: List,
                                 del_index: int) \
        -> (List, List):
    """
    Deletes a Row from the Option + Constrains and Rebuilds all Choices for Variables
    :param model: the JPT model of the Prob. Tree
    :param variable_div: list of Components to Chose Variable in the GUI
    :param constrains_div: list of Components that are the Constraints for the Variables on the Same Index
    :param del_index: the Value on what Position the to delete Row is.
    :return: Variable Children and Constrains Children for the GUI withe Update options
    """
    variable_list = variable_div
    constrains_list = constrains_div
    option_list = option_div

    variable_list.pop(del_index)
    constrains_list.pop(del_index)
    option_list.pop(del_index)
    new_var_list = c.update_free_vars_in_div(model, variable_list)
    option_list[-1]['props']['disabled'] = True
    return new_var_list, constrains_list, option_list


def correct_input_div(variable, value):
    if variable.numeric:
        minimum = priors[variable].cdf.intervals[0].upper
        maximum = priors[variable].cdf.intervals[-1].lower
        rang = c.create_range_slider(minimum, maximum, id={'type': 'op_i', 'index': 0}, value=value, dots=False, tooltip={"placement": "bottom", "always_visible": False})
        return rang
    else:
        return dcc.Dropdown(id={'type': 'op_i', 'index': 0}, options=value, value=value)


def generate_modal_option(model, var, value):
    modal_layout = []
    modal_layout.append(dbc.ModalHeader(dbc.ModalTitle(var)))
    variable = model.varnames[var]
    result = model.posterior(evidence={})
    print(f"value= {value}")

    body = dbc.ModalBody([
        dbc.Row([  # Grapicen
            dbc.Col([
                c.plot_numeric_to_div(var, result) if variable.numeric else c.plot_symbolic_to_div(var, result)
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([  # Inputs
                html.Div(correct_input_div(variable, value), id="mod_in"),
                #html.Div([dcc.RangeSlider(id={'type': 'op_i', 'index': 0}, min=0, max=10, value=[2, 6], dots=False, tooltip={"placement": "bottom", "always_visible": False})], id="mod_in"),
            ], width=6, className="d-grid ps-2")
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Button("+", id="op_add", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0)
            ], width=6, className="d-grid ps2")
        ])
    ])

    foot = dbc.ModalFooter(children=[
        dbc.Button("Abort", id=dict(type="option_abort", index=0), className="ms-auto", n_clicks=0),
        dbc.Button("Save", id=dict(type="option_save", index=0), className="ms-auto", n_clicks=0)
    ])
    modal_layout.append(body)
    modal_layout.append(foot)
    return modal_layout


@app.callback(
    Output('e_variable', 'children'),
    Output('e_input', 'children'),
    Output('e_option', 'children'),
    Output('modal_option', 'children'),
    Output('modal_option', 'is_open'),
    Input("upload_tree", 'contents'),
    Input({'type': 'dd_e', 'index': ALL}, 'value'),
    Input({'type': 'b_e', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_save', 'index': ALL}, 'n_clicks'),
    Input({'type': 'option_abort', 'index': ALL}, 'n_clicks'),
    State('e_variable', 'children'),
    State('e_input', 'children'),
    State('e_option', 'children'),
    State({'type': 'op_i', 'index': ALL}, 'value'),
)
def evid_gen(upload, dd_vals, b_e, op_s, op_a, e_var, e_in, e_op, op_i):
    global modal_basic
    e_var: List[dict] = e_var
    e_in: List[dict] = e_in
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
            return e_var, e_in, e_op, modal_basic, False
        e_var_n, e_in_n, e_op_n = reset_gui_button(io_model, "e")
        model = io_model
        priors = model.independent_marginals()
        return e_var_n, e_in_n, e_op_n, modal_basic, False
    elif cb.get("type") == "dd_e":
        if dd_vals[cb.get("index")] is None:
            return *del_selector_from_div_button(model, e_var, e_in, e_op, cb.get("index")), [], False
        variable = model.varnames[dd_vals[cb.get("index")]]
        if variable.numeric:
            minimum = priors[variable].cdf.intervals[0].upper
            maximum = priors[variable].cdf.intervals[-1].lower
            e_in[cb.get("index")] = c.create_range_slider(minimum, maximum,
                                                          id={'type': 'i_e', 'index': cb.get("index")},
                                                          value=[minimum, maximum],
                                                          dots=False,
                                                          tooltip={"placement": "bottom", "always_visible": False})

        elif variable.symbolic:
            e_in[cb.get("index")] = dcc.Dropdown(id={"type": "i_e", "index": cb.get("index")},
                                                 options={k: v for k, v in zip(variable.domain.labels.values(),
                                                                               variable.domain.labels.values())},
                                                 value=list(variable.domain.labels.values()),
                                                 multi=True, )

        if len(e_var) - 1 == cb.get("index"):
            return *add_selector_to_div_button(model, e_var, e_in, e_op, "e", cb.get("index") + 1), [], False
    elif cb.get("type") == "b_e" and dd_vals[cb.get("index")] != []:
        # Dont Like dont know to do it other wise
        global modal_var_index
        modal_var_index = cb.get("index")
        modal_body = generate_modal_option(model=model, var=e_var[cb.get("index")]['props']['value'],
                                           value=e_in[cb.get("index")]['props']['drag_value'])
        return e_var, e_in, e_op, modal_body, True
    elif cb.get("type") == "option_save":
        new_vals = []
        #new_list = []
        sor_val = sorted(op_i, key=lambda x: x[0])
        while sor_val != []:
            if len(sor_val) > 1 and sor_val[0][1] >= sor_val[1][0]:
                if sor_val[0][1] >= sor_val[1][1]:
                    sor_val.pop(1)
                else:
                    sor_val[0] = [sor_val[0][0], sor_val[1][1]]
                    sor_val.pop(1)
            else:
                new_vals.append(sor_val[0][0])
                new_vals.append(sor_val[0][1])

                #new_list.append(sor_val[0])
                sor_val.pop(0)

        e_in[modal_var_index]['props']['value'] = new_vals
        e_in[modal_var_index]['props']['drag_value'] = new_vals
        return e_var, e_in, e_op, [], False
    # "option_abort kann als Default gewertet werden, da es keinen Inpakt hat."
    return c.update_free_vars_in_div(model, e_var), e_in, e_op, [], False


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
    print(m_in_new)
    print("-----")
    return m_in_new


if __name__ == '__main__':
    app.run_server(debug=True)

#CSS SLIDER
#Logik für Combenieren