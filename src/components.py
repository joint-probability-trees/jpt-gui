import math
from dash import dcc, html
import plotly.graph_objects as go
import jpt.variables
import jpt.base.intervals
import dash_bootstrap_components as dbc
from typing import List
import os
import jpt.base.utils
from jpt.base.utils import list2interval
import numpy as np

in_use_tree = jpt.JPT([jpt.variables.NumericVariable("")])

priors = None



color_list_modal = ["#ccff66", "MediumSeaGreen", "Tomato", "SlateBlue", "Violet"]

# default_tree.varnames
# default_tree.features
# default_tree.targets
# FRontpage alle werte
# Nav
# TItle (Name der Datei)
# Number of Paras
# List Varnames in Farben

#LAden BUtten TaskLeiste Breiter machen Button fixen Home func schrieben

# ---MODAL_EYE____
def gen_modal_basic_id(id: str):
    """
        Generates the zoom Modal style Basic Dash Objects withe the Correct ID
    :param id: The Id to Specify the Components for Dash Callbacks
    :return: Zoom Modal Dash Object List
    """
    return [
        dbc.ModalHeader(dbc.ModalTitle('temp')),
        dbc.ModalBody([
            html.Div([dcc.Dropdown(id={'type': f'op_i{id}', 'index': 0}), dbc.Button(id=f"op_add{id}")], id="mod_in")
        ]),
        dbc.ModalFooter(
            [
                dbc.Button("Save", id=dict(type=f"option_save{id}", index=0), className="ms-auto", n_clicks=0)
            ]
        ),
    ]
def gen_modal_option_id(id: str):
    """
        Generates the Zoom Modal Obtions where the Inteative Components will be set
    :param id: The Id to Specify the Components for Dash Callbacks
    :return: Modal Components withe the base for the Inteactive parts
    """
    return dbc.Modal(
        [
            # #Chidlren? alles Generieren
            dbc.ModalHeader(dbc.ModalTitle('temp'), id="mod_header"),
            dbc.ModalBody([

                dbc.Row(id=f"modal_input{id}", children=[
                    dbc.Col([], id={'type': f'op_i{id}', 'index': 0},
                            className="d-flex flex-nowrap justify-content-center ps-2")
                ], className="d-flex justify-content-center"),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("+", id=f"op_add{id}", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0,
                                   disabled=True)
                    ], width=6, className="d-grid ps2")
                ]),
                dbc.ModalFooter(
                    [
                        dbc.Button("Save", id=dict(type=f"option_save{id}", index=0), className="ms-auto", n_clicks=0)
                    ]
                ),
            ],)
        ],
        id=f"modal_option{id}", is_open=False, size="xl", backdrop="static"
    )


# ---/MODAL-EYE---
# --- MODAL-FUNC ---
def correct_input_div(variable, value, priors, id, **kwargs):
    """
        Generate a Dash Componant for the Varibael, that can be used in the zoom Modal
    :param variable: The Variabel wich is be displayed
    :param value:  The Value of the Variable chosen from the User
    :param priors: the Priors of the modael
    :param kwargs: further specifation for the Dash Componant
    :return: a Dash Componant that displays the variable
    """
    if variable.numeric:
        minimum = priors[variable.name].cdf.intervals[0].upper
        maximum = priors[variable.name].cdf.intervals[-1].lower
        rang = create_range_slider(minimum, maximum, id={'type': f'op_i{id}', 'index': 0}, value=value, dots=False,
                                   tooltip={"placement": "bottom", "always_visible": False}, **kwargs)
        return rang
    elif variable.symbolic:
        return dcc.Dropdown(id={'type': f'op_i{id}', 'index': 0},
                            options={k: v for k, v in zip(variable.domain.labels.values(), variable.domain.labels.values())},
                            value=value, multi=True, **kwargs)
    elif variable.integer:
        lab = list(variable.domain.labels.values())
        mini = min(lab)
        maxi = max(lab)
        markings = dict(zip(lab, map(str, lab)))
        return create_range_slider(minimum=mini, maximum=maxi, value=[mini, maxi],
                                                      id={'type': f'op_i{id}', 'index': 0}, dots=False,
                                                      marks=markings,
                                                      tooltip={"placement": "bottom", "always_visible": False}, **kwargs)


def generate_correct_plots(variable, var, result):
    if variable.numeric:
        return plot_numeric_to_div(var, result=result)
    elif variable.symbolic:
        return plot_symbolic_to_div(var, result=result)
    elif variable.integer:
        return plot_symbolic_to_div(var, result=result)

def generate_modal_option(model: jpt.trees.JPT, var: str, value: List[str or int or float], priors, id):
    """
        Creates a modal for Zoom for a chosen Variabel, the Style is static
    :param model: the model of the Tree
    :param var: the Variabel wiche will be displayed
    :param value: the User chosen Values from the Varibale
    :param priors: the Priors pre calculatet
    :param id: id from Modal will be modal_input_id because the callbacks cant be duplicated
    :return: Zoom Modal for the Variabel in var
    """
    modal_layout = []
    modal_layout.append(dbc.ModalHeader(dbc.ModalTitle(var)))
    variable = model.varnames[var]
    result = model.posterior(evidence={})
    map = div_to_variablemap(model, [var], [value])
    probs = model.infer(map, {})

    body = dbc.ModalBody(id=f"modal_input{id}", children=[
        dbc.Row([  # Grapicen
            dbc.Col([
                generate_correct_plots(variable, var, result)
            ], width=12),
        ]),
        dbc.Row(children=[
            html.Div([  # Inputs
                html.Div("Range 1" if variable.numeric or variable.integer else "Dropmenu", style=dict(color=color_list_modal[0])),
                correct_input_div(variable, value, priors=priors, id=id ,className="d-flex flex-fill"),
                html.Div(f"{probs}", style=dict(color=color_list_modal[0])),
            ], id="modal_color_0", className="d-flex justify-content-evenly ps-2")
        ],className="d-flex justify-content-evenly"),
        dbc.Row([
            dbc.Col([
                dbc.Button("+", id=f"op_add{id}", className="d-grid gap-2 col-3 mt-3 mx-auto", n_clicks=0,
                           disabled=True if variable.symbolic else False)
            ], width=6, className="d-grid ps2")
        ])
    ])

    foot = dbc.ModalFooter(children=[
        dbc.Button("Save", id=dict(type=f"option_save{id}", index=0), className="ms-auto", n_clicks=0)
    ])
    modal_layout.append(body)
    modal_layout.append(foot)
    return modal_layout


# --- /MODAL_FUNC ---


def create_range_slider(minimum: float, maximum: float, *args, **kwargs) -> \
        dcc.RangeSlider:
    """
        Generate a RangeSlider that resembles a continuous set.
    :param minimum: lowest number possible in the Range of the slider (left-Side)
    :param maximum: the Highest number possible in the Range of the slider (right-Side)
    :param args: Further styling for plotly dash components
    :param kwargs: Further styling for plotly dash components
    :return: The slider as dcc component
    """

    if minimum == maximum:
        minimum -= 1
        maximum += 1

    slider = dcc.RangeSlider(**kwargs, min=math.floor(minimum), max=math.ceil(maximum), allowCross=False)

    return slider


def fuse_overlapping_range(ranges: List) -> List:
    new_vals = []
    new_list = []
    sor_val = sorted(ranges, key=lambda x: x[0])
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

            new_list.append(sor_val[0])
            sor_val.pop(0)
    return new_vals


def div_to_variablemap(model: jpt.trees.JPT, variables: List, constrains: List) -> jpt.variables.VariableMap:
    """
        Transforms variable and Constrains List form the GUI to a VariableMap
    :param model: the JPT model of the Prob. Tree
    :param variables: The list of chosen Variables
    :param constrains:  The list of for the Variables on the same Index
    :return: VariableMap of the Variables with its associated Constraints
    """
    var_dict = {}

    for variable, constrain in zip(variables, constrains):
        if variable is None or constrain is None:
            continue

        if model.varnames[variable].numeric:
            var_dict[variable] = jpt.base.intervals.ContinuousSet(constrain[0], constrain[1])
        elif model.varnames[variable].integer:
            var_dict[variable] = set([round(x) for x in constrain])
        else:
            var_dict[variable] = set(constrain)

    return model.bind(var_dict)
    #return jpt.variables.VariableMap([(model.varnames[k], v) for k, v in var_dict.items()])


def mpe_result_to_div(model: jpt.trees.JPT, res: jpt.trees.VariableMap, likelihood: float) -> List:
    """
        Generate Visuel Dash Representation for result of the mpe jpt func
    :param res: one of the Results from mpe func
    :param likelihood: The likelihood of the maxima
    :return: Children's List from Dash Components to display the Results in res
    """
    return_div = []

    for variable, restriction in res.items():
        if variable.integer:
            value = [x for i in range(0, len(restriction)) for x in (i, i)]
            lab = list(variable.domain.labels.values())
            mini = min(lab)
            maxi = max(lab)
            markings = dict(zip(lab, map(str, lab)))
            return_div += [html.Div(
                [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True, className="margin10"),
                 create_range_slider(minimum=mini-1, maximum=maxi+1, value=value, disabled=True, marks=markings, dots=False,
                                     className="margin10")]
                , style={"display": "grid", "grid-template-columns": "30% 70%"})]

        if variable.numeric:
            value = []
            if type(res[variable]) == jpt.base.intervals.RealSet:
                for interval in res[variable].intervals:
                    value += [interval.lower, interval.upper]
            else:
                value += [res[variable].lower, res[variable].upper]

            minimum = model.priors[variable.name].cdf.intervals[0].upper
            maximum = model.priors[variable.name].cdf.intervals[-1].lower
            return_div += [html.Div(
                [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True, className="margin10"),
                 create_range_slider(minimum, maximum, value=value, disabled=True, className="margin10",
                                     tooltip={"placement": "bottom", "always_visible": True})]
                , style={"display": "grid", "grid-template-columns": "30% 70%"})]
        elif variable.symbolic:
            return_div += [html.Div(
                [dcc.Dropdown(options=[variable.name], value=variable.name, disabled=True),
                 dcc.Dropdown(
                     options=list(restriction),
                     value=list(restriction), multi=True, disabled=True, className="ps-3")],
                style={"display": "grid", "grid-template-columns": "30% 70%"})]
        return_div += [html.Div(className="pt-1")]

    return_div = [html.Div([dcc.Dropdown(options=["Likelihood"], value="Likelihood", disabled=True,
                                         className="margin10"),
                            dcc.Dropdown(options=[likelihood], value=likelihood, disabled=True, className="ps-3 pb-2")],
                           id="likelihood", style={"display": "grid", "grid-template-columns": "30% 70%"})] + return_div

    return return_div


def create_prefix_text_query(len_fac_q: int, len_fac_e: int) -> List:
    """
        Creates Dash Style Prefix for the query GUI
    :param len_fac_q:  Length of Query input used for Scaling
    :param len_fac_e:  Length of Evidence input used for Scaling
    :return: Children div for the prefix query GUI
    """
    return [
        html.Div("P ", className="pe-3",
                 style={"width": "50%", "height": "100%",
                        'fontSize': (len_fac_q if len_fac_q >= len_fac_e else len_fac_e) * 20,
                        'padding-top': (len_fac_q * 1 if len_fac_q >= len_fac_e else len_fac_e * 1)}),
    ]


def create_prefix_text_mpe(len_fac: int) -> List:
    """
        Creates Dash Style Prefix for the MPE GUI
    :param len_fac: Length of Evidence input used for Scaling
    :return: Children div for the prefix MPE GUI
    """
    return [
        html.Div("argmax ", className="pe-3",
                 style={'padding-top': 0, 'fontSize': len_fac * 10 if len_fac * 10 < 40 else 25}),
        html.Div("P ", className="ps-3",
                 style={'padding-top': 0, "height": "100%", 'fontSize': len_fac * 15 if len_fac * 15 < 75 else 75}),
    ]


def generate_free_variables_from_div(model: jpt.trees.JPT, variable_div: List) -> List[str]:
    """
        Peels the names out of variable_div elements and uses generate_free_variables_from_list for the Return
    :param model: the JPT model of the Prob. Tree
    :param variable_div: List of all Variabels that are being Used, in Dash Dropdown Class saved
    :return: Returns List of String from the Names of all not used Variabels.
    """
    variable_list = variable_div

    variables = []
    for v in variable_list:
        if len(v['props']) > 2:
            variables += [v['props'].get('value', [])]
    return generate_free_variables_from_list(model, variables)


def generate_free_variables_from_list(model: jpt.trees.JPT, variable_list: List[str]) -> List[str]:
    """
        Deletes all used Variable Names out of a List of all Variables Names.
    :param model: the JPT model of the Prob. Tree
    :param variable_list: the List of in use Variable Names
    :return: List of Variable Names that are not in use
    """
    vars_free = model.varnames.copy()

    for v in variable_list:
        if v != []:
            vars_free.pop(v)
    return list(vars_free.keys())


def update_free_vars_in_div(model: jpt.trees.JPT, variable_div: List) -> List:
    """
        Updates the Variable Options for a Dash Dropdown for choosing Variables, to all not in use Variables.
    :param model: the JPT model of the Prob. Tree
    :param variable_div: the Div to update the Options
    :return: the Div withe updated variable Options
    """
    variable_list = variable_div
    vars_free = generate_free_variables_from_div(model, variable_list)
    d = dict(a="a", b="b")

    for v in variable_list:
        if len(v['props']) > 2:
            if v['props'].get('value', "NULL") == "NULL":
                v['props']['options'] = vars_free
            else:
                v['props']['options'] = [v['props'].get('value')] + vars_free
    return variable_list


def reduce_index(index, number, list) -> List:
    """
        Reduces the index in id from index in the list about the amount number
    :param index: the start index to decrease the index
    :param number: the amount to decrease
    :param list: the List from Dash Components that should be decreased
    :return: list with the decreased index implemented
    """
    for i in range(index, len(list)):
        list[i]['props']['id']['index'] -= number
    return list


def del_selector_from_div(model: jpt.trees.JPT, variable_div: List, constrains_div: List, del_index: int) \
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

    variable_list = reduce_index(del_index, 1, variable_list)
    constrains_list = reduce_index(del_index, 1, constrains_list)

    variable_list.pop(del_index)
    constrains_list.pop(del_index)

    new_var_list = update_free_vars_in_div(model, variable_list)
    return new_var_list, constrains_list

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

    # if len(variable_list) == 1:
    #     variable_list[0]['props']['value'] = ""
    # else:

    variable_list = reduce_index(del_index, 1, variable_list)
    constrains_list = reduce_index(del_index, 1, constrains_list)
    option_list = reduce_index(del_index, 1, option_list)

    variable_list.pop(del_index)
    constrains_list.pop(del_index)
    option_list.pop(del_index)

    new_var_list = update_free_vars_in_div(model, variable_list)
    option_list[-1]['props']['disabled'] = True
    return new_var_list, constrains_list, option_list


def add_selector_to_div(model: jpt.trees.JPT, variable_div: List, constrains_div: list, type: str,
                        index: int) \
        -> (List[dcc.Dropdown], List):
    """
        Genrats the Correct Selector Components for the div
    :param model: the JPT model of the Prob. Tree
    :param variable_div: list of Components to Chose Variable in the GUI
    :param constrains_div: list of Components that are the Constraints for the Variables on the Same Index
    :param type: the Type of the Component for the ID
    :param index: the index Number of the Component for the ID
    :return: Variable Children and Constrains Children for the GUI withe one more Row
    """
    variable_list = variable_div
    constrains_list = constrains_div

    variable_list = update_free_vars_in_div(model, variable_list)

    variable_list.append(
        dcc.Dropdown(id={'type': f'dd_{type}', 'index': index},
                     options=variable_list[0]['props']['options'][1:]))
    constrains_list.append(dcc.Dropdown(id={'type': f'i_{type}', 'index': index}, disabled=True))
    return variable_list, constrains_list


# --- Button Func ---
def add_selector_to_div_button(model: jpt.trees.JPT, variable_div, constrains_div, option_div, type: str,
                               index: int) \
        -> (List[dcc.Dropdown], List, List):
    """
        Genrates teh Selector for the div withe a Button
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

    variable_list = update_free_vars_in_div(model, variable_list)
    option_list[-1]['props']['disabled'] = False

    variable_list.append(
        dcc.Dropdown(id={'type': f'dd_{type}', 'index': index},
                     options=variable_list[0]['props']['options'][1:], className=""))
    constrains_list.append(dcc.Dropdown(id={'type': f'i_{type}', 'index': index}, disabled=True, className="", style={'padding-top': 0}))
    option_list.append(
        dbc.Button("👁️", id=dict(type=f'b_{type}', index=index), disabled=True, n_clicks=0, className="",
                   size="sm"))
    return variable_list, constrains_list, option_list


def reset_gui_button(model: jpt.trees.JPT, type: str):
    """
        Resets the GUI Parts back to Start + Button
    :param model: the JPT Tree
    :param type: What Type of ID it is
    :return: Clean Start Style of Components for the GUI
    """
    var_div = [dcc.Dropdown(id={'type': f'dd_{type}', 'index': 0}, options=sorted(model.varnames))]
    in_div = [dcc.Dropdown(id={'type': f'i_{type}', 'index': 0}, disabled=True)]
    op_div = [dbc.Button("👁️", id=dict(type='b_e', index=0), disabled=True, n_clicks=0, className="me-2",
                         size="sm")]
    return var_div, in_div, op_div


# --- Button Func ---


def reset_gui(model: jpt.trees.JPT, type: str) -> (List, List):
    """
        Resets the GUI Parts back to Start
    :param model: the JPT Tree
    :param type: What Type of ID it is
    :return: Clean Start Style of Components for the GUI
    """
    var_div = [dcc.Dropdown(id={'type': f'dd_{type}', 'index': 0}, options=sorted(model.varnames))]
    in_div = [dcc.Dropdown(id={'type': f'i_{type}', 'index': 0}, disabled=True)]
    return var_div, in_div


# Postierior---

def plot_symbolic_distribution(distribution: jpt.distributions.univariate.Multinomial) -> go.Bar:
    """
        generates a Bar graph for symbolic distribution in jpt.
    :param distribution: the Distribution for the Bar Diagram
    :return: the trace of a Bar Diagram for the symbolic variable.
    """
    trace = go.Bar(x=list(distribution.labels.keys()), y=distribution._params)  # anstatt keys könnte values sein
    return trace

#TODOO
#X nach Externe Konvertierenn
def plot_numeric_pdf(distribution: jpt.distributions.univariate.Numeric, padding=0.1) -> go.Scatter:
    """
        generates a jpt plot from a numeric variable
    :param distribution: the Distribution of the variable for the Plot
    :param padding: for the ends of the Plot, it is for visibility.
    :return: scatter plot for the numeric variable
    """
    x = []
    y = []

    for interval, function in zip(distribution.pdf.intervals[1:-1], distribution.pdf.functions[1:-1]):
        x += [interval.lower, interval.upper, interval.upper]
        y += [function.value, function.value, None]

    x = [distribution.value2label(x_) for x_ in x ]
    range = x[-1] - x[0]
    x = [x[0] - (range * padding), x[0], x[0]] + x + [x[-1], x[-1], x[-1] + (range * padding)]
    y = [0, 0, None] + y + [None, 0, 0]

    trace = go.Scatter(x=x, y=y, name="PDF")

    # generate logarithmic scaled trace
    log_y = [np.log(y_) if y_ is not None and y_ > 0 else None for y_ in y]
    log_trace = go.Scatter(x=x, y=log_y, name="Logarithmic PDF", visible='legendonly')

    return trace, log_trace


def plot_numeric_cdf(distribution: jpt.distributions.univariate.Numeric, padding=0.1) -> go.Scatter:
    """
        generates a cdf plot from a numeric variable
    :param distribution: the Distribution of the variable for the Plot
    :param padding: for the ends of the Plot, it is for visibility.
    :return: scatter plot for the numeric variable
    """
    x = []
    y = []

    for interval, function in zip(distribution.cdf.intervals[1:], distribution.cdf.functions[1:]):
        x += [interval.lower]
        y += [function.eval(interval.lower)]

    x = [distribution.value2label(x_) for x_ in x ]
    range = x[-1] - x[0]
    if range == 0:
        range = 1

    x = [x[0] - (range * padding), x[0]] + x + [x[-1] + (range * padding)]
    y = [0, 0] + y + [1]
    trace = go.Scatter(x=x, y=y, name="CDF")
    return trace


def plot_numeric_to_div(var_name: List, result) -> List:
    """
        Generates a Div where both plots are in for a numeric variable
    :param var_name: the name of variable that will be plotted
    :param result: the result generate from jpt.
    :return: one div withe 2 Plots in.
    """
    fig = go.Figure(layout=dict(title=f"Cumulative Density Function of {var_name}"))
    t = plot_numeric_cdf(result[var_name])
    fig.add_trace(t)
    is_dirac = result[var_name].is_dirac_impulse()
    if not is_dirac:
        fig2 = go.Figure(layout=dict(title=f"Probability Density Function of {var_name}"))
        t2, t3 = plot_numeric_pdf(result[var_name])
        fig2.add_trace(t2)
        fig2.add_trace(t3)



    arg_max, max_ = result[var_name].mpe()
    arg_max = result[var_name].value2label(arg_max)

    arg_max = arg_max.simplify()
    if isinstance(arg_max, jpt.base.intervals.ContinuousSet):
        arg_max = jpt.base.intervals.RealSet([arg_max])

    for interval in arg_max.intervals:
        if interval.size() <= 1:
            continue

        fig.add_trace(go.Scatter(x=[interval.lower, interval.upper, interval.upper, interval.lower],
                                 y=[0, 0,result[var_name].p(list2interval([-float("inf"), interval.upper])),
                                    result[var_name].p(list2interval([-float("inf"), interval.lower]))],
                                 fillcolor="LightSalmon",
                                 opacity=0.5,
                                 mode="lines",
                                 fill="toself", line=dict(width=0),
                                 name="Max"))
        if not is_dirac:
            fig2.add_trace(go.Scatter(x=[interval.lower, interval.upper, interval.upper, interval.lower],
                                      y=[0, 0, max_, max_],
                                      fillcolor="LightSalmon",
                                      opacity=0.5,
                                      mode="lines",
                                      fill="toself", line=dict(width=0),
                                      name="Max"))

    try:
        expectation = result[var_name].expectation()
        fig.add_trace(go.Scatter(x=[expectation, expectation], y=[0, 1], name="Exp", mode="lines+markers",
                                marker=dict(opacity=[0, 1])))
    except:
        pass

    if is_dirac:
        return html.Div([dcc.Graph(figure=fig), html.Div(className="pt-2")], className="pb-3")
    else:
        try:
            expectation = result[var_name].expectation()
            fig2.add_trace(go.Scatter(x=[expectation, expectation], y=[0, max_ * 1.1], name="Exp", mode="lines+markers",
                                  marker=dict(opacity=[0, 1])))
        except:
            pass
        return html.Div([dcc.Graph(figure=fig), html.Div(className="pt-2"), dcc.Graph(figure=fig2)], className="pb-3")


def plot_symbolic_to_div(var_name: str, result) -> List:
    """
        generates a div where a bar Diagram for a Symbolic Variable.
    :param var_name: the name of the variable
    :param result: the result generate from jpt
    :return: a div withe one bar diagram in it.
    """
    max_, arg_max = result[var_name].mpe()
    fig = go.Figure(layout=dict(title="Probability Distribution"))
    lis_x_max = []
    lis_y_max = []
    lis_x = []
    lis_y = []
    for i in range(0, len(result[var_name].labels.keys())):
        if result[var_name]._params[i] >= max_:
            lis_x_max += [list(result[var_name].labels.keys())[i]]
            lis_y_max += [result[var_name]._params[i]]
        else:
            lis_x += [list(result[var_name].labels.keys())[i]]
            lis_y += [result[var_name]._params[i]]

    lis_x = [result[var_name].value2label(x_) for x_ in lis_x]
    lis_x_max = [result[var_name].value2label(x_) for x_ in lis_x_max]

    fig.add_trace(go.Bar(y=lis_x_max, x=lis_y_max, name="Max", marker=dict(color="LightSalmon"), orientation="h"))
    fig.add_trace(go.Bar(y=lis_x, x=lis_y, name="Prob", marker=dict(color="CornflowerBlue"), orientation='h',))
    return html.Div([dcc.Graph(figure=fig)], className="pb-3")


def gen_Nav_pages(pages, toIgnoreName):
    """
        Genartes the Navigation Page Links, withe out the toIgnoreNames
    :param pages: All Pages that are in the GUI
    :param toIgnoreName: Names of Pages that shouldnt be displayed (Empty)
    :return: Dash Struct for Navgation of Pages
    """
    nav = [p for p in pages if p['name'].lower() not in [x.lower() for x in toIgnoreName]]
    nav_posi = dict(Home = 0, Query = 1, Most_Probable_Explanation = 3, Posterior = 2)
    navs = oder_Nav(nav_posi, nav)
    navItems = []
    for page in navs:
        navItems.append(dbc.NavItem(dbc.NavLink(f"{page['name']}", href=page['relative_path'])))
        #Liste solle Home Query Most probable explanation Posterior Rest sein

    return navItems

def oder_Nav(nav_positions: dict, nav : List):
    # sollte in Kontext gehen ohne wieder holte sortieren
    sor = True
    while sor:
        sor = False
        for index, n in enumerate(nav):
            posi = nav_positions.get(n['name'], -1)
            if posi != index and posi != -1:
                sor = True
                nav[posi], nav[index] = nav[index], nav[posi]

    return nav

