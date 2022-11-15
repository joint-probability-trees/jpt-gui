import math
from dash import dcc, html
import plotly.graph_objects as go
import jpt.variables
import jpt.base.intervals
from typing import List


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

    if min == max:
        minimum = min - 1
        maximum = max + 1

    slider = dcc.RangeSlider(**kwargs, min=math.floor(minimum), max=math.ceil(maximum), allowCross=False)

    return slider


def real_set_to_rangeslider(realset: jpt.base.intervals.RealSet) -> dcc.RangeSlider:
    pass


def div_to_variablemap(model: jpt.trees.JPT, variables: List, constrains: List) -> jpt.variables.VariableMap:
    """
    Transforms variable and Constrains List form the GUI to a VariableMap
    :param model: the JPT model of the Prob. Tree
    :param variables: The list of chosen Variables
    :param constrains:  The list of for the Variables on the same Index
    :return: VariableMap of the Variables with its associated Constraints
    """
    var_dict = {}
    print(f'vars:{variables}  , cons{constrains}')
    for variable, constrain in zip(variables, constrains):
        if variable is None or constrain is None:  # TODOO WIESO HAT DAS NULLS
            continue

        if model.varnames[variable].numeric:
            var_dict[variable] = jpt.base.intervals.ContinuousSet(constrain[0], constrain[1])
        else:
            var_dict[variable] = set(constrain)

    return jpt.variables.VariableMap([(model.varnames[k], v) for k, v in var_dict.items()])


def mpe_result_to_div(model: jpt.trees.JPT, res: List[jpt.trees.MPEResult]) -> List:
    """
        Generate Visuel Dash Representation for result of the mpe jpt func
    :param res: one of the Results from mpe func
    :return: Children's List from Dash Components to display the Results in res
    """
    return_div = []

    for variable, restriction in res.maximum.items():

        if variable.numeric:
            value = []
            for interval in res.maximum[variable].intervals:
                value += [interval.lower, interval.upper]

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
                            dcc.Dropdown(options=[res.result], value=res.result, disabled=True, className="ps-3 pb-2")],
                           id="likelihood", style={"display": "grid", "grid-template-columns": "30% 70%"})] + return_div

    return return_div


def create_prefix_text_query(len_fac_q: List, len_fac_e: int) -> List:
    """
    Creates Dash Style Prefix for the query GUI
    :param len_fac_q:  Length of Query input used for Scaling
    :param len_fac_e:  Length of Evidence input used for Scaling
    :return: Children div for the prefix query GUI
    """
    return [
        html.Div("P ", className="align-self-center text-end float-end",
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
                 style={'padding-top': 0, 'fontSize': len_fac * 10 if len_fac * 10 < 40 else 40}),
        html.Div("P ", className="ps-3",
                 style={'padding-top': 0, "height": "100%", 'fontSize': len_fac * 15 if len_fac * 15 < 75 else 75}),
    ]


def generate_free_variables_from_div(model: jpt.trees.JPT, variable_div: List) -> List:
    variable_list = variable_div
    variables = []
    for v in variable_list:
        if len(v['props']) > 2:
            print(v)
            variables += [v['props']['value']]
    return generate_free_variables_from_list(model, variables)


def generate_free_variables_from_list(model: jpt.trees.JPT, variable_list: List) -> List:
    vars_free = model.varnames.copy()
    # return [v for v in model.varnames if v not in varaible_list]
    print(variable_list)
    for v in variable_list: vars_free.pop(v)
    return list(vars_free.keys())


def update_free_vars_in_div(model: jpt.trees.JPT, variable_div: List) -> List:
    variable_list = variable_div
    vars_free = generate_free_variables_from_div(model, variable_list)

    for v in variable_list:
        if len(v['props']) > 2:
            print(v)
            v['props']['options'] = [v['props']['value']] + vars_free

    return variable_list

def reduce_index(index, number, list):
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


def add_selector_to_div(model: jpt.trees.JPT, variable_div: List, constrains_div: list, type: str,
                        index: int) \
        -> (List[dcc.Dropdown], List):
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

    variable_list = update_free_vars_in_div(model, variable_list)

    variable_list.append(
        dcc.Dropdown(id={'type': f'dd_{type}', 'index': index},
                     options=variable_list[0]['props']['options'][1:]))
    constrains_list.append(dcc.Dropdown(id={'type': f'i_{type}', 'index': index}, disabled=True))
    return variable_list, constrains_list


def reset_gui(model: jpt.trees.JPT, type: str) -> (List, List):
    var_div = [dcc.Dropdown(id={'type': f'dd_{type}', 'index': 0}, options=sorted(model.varnames))]
    in_div = [dcc.Dropdown(id={'type': f'i_{type}', 'index': 0}, disabled=True)]
    return var_div, in_div


# Postierior---

def plot_symbolic_distribution(distribution: jpt.distributions.univariate.Multinomial) -> go.Bar:
    trace = go.Bar(x=list(distribution.labels.keys()), y=distribution._params)  # anstatt keys könnte values sein
    return trace


def plot_numeric_pdf(distribution: jpt.distributions.univariate.Numeric, padding=0.1) -> go.Scatter:
    x = []
    y = []
    for interval, function in zip(distribution.pdf.intervals[1:-1], distribution.pdf.functions[1:-1]):
        x += [interval.lower, interval.upper, interval.upper]
        y += [function.value, function.value, None]

    range = x[-1] - x[0]
    x = [x[0] - (range * padding), x[0], x[0]] + x + [x[-1], x[-1], x[-1] + (range * padding)]
    y = [0, 0, None] + y + [None, 0, 0]
    trace = go.Scatter(x=x, y=y, name="PDF")
    return trace


def plot_numeric_cdf(distribution: jpt.distributions.univariate.Numeric, padding=0.1) -> go.Scatter:
    x = []
    y = []

    for interval, function in zip(distribution.cdf.intervals[1:], distribution.cdf.functions[1:]):
        x += [interval.lower]
        y += [function.eval(interval.lower)]

    range = x[-1] - x[0]
    if range == 0:
        range = 1

    x = [x[0] - (range * padding), x[0]] + x + [x[-1] + (range * padding)]
    y = [0, 0] + y + [1]
    trace = go.Scatter(x=x, y=y, name="CDF")
    return trace


def plot_numeric_to_div(var_name: List, result) -> List:
    fig = go.Figure(layout=dict(title=f"Cumulative Density Function of {var_name}"))
    t = plot_numeric_cdf(result[var_name])
    fig.add_trace(t)
    is_dirac = result[var_name].is_dirac_impulse()
    if not is_dirac:
        fig2 = go.Figure(layout=dict(title=f"Probability Density Function of {var_name}"))
        t2 = plot_numeric_pdf(result[var_name])
        fig2.add_trace(t2)

    expectation = result[var_name].expectation()
    max_, arg_max = result[var_name].mpe()

    for interval in arg_max.intervals:
        if interval.size() <= 1:
            continue
        fig.add_trace(go.Scatter(x=[interval.lower, interval.upper, interval.upper, interval.lower],
                                 y=[0, 0, result[var_name].cdf.eval(interval.upper),
                                    result[var_name].cdf.eval(interval.lower)],
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
    fig.add_trace(go.Scatter(x=[expectation, expectation], y=[0, 1], name="Exp", mode="lines+markers",
                             marker=dict(opacity=[0, 1])))
    if is_dirac:
        return html.Div([dcc.Graph(figure=fig), html.Div(className="pt-2")],className="pb-3")
    else:
        fig2.add_trace(go.Scatter(x=[expectation, expectation], y=[0, max_ * 1.1], name="Exp", mode="lines+markers",
                                  marker=dict(opacity=[0, 1])))
        return html.Div([dcc.Graph(figure=fig), html.Div(className="pt-2"), dcc.Graph(figure=fig2)],className="pb-3")


def plot_symbolic_to_div(var_name: str, result) -> List:
    max_, arg_max = result[var_name].mpe()
    fig = go.Figure(layout=dict(title="Probability Distribution"))
    # t = plot_symbolic_distribution(result[var_name])
    # fig.add_trace(t)
    # trace = go.Bar(x=list(distribution.labels.keys()), y=distribution._params)
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

    fig.add_trace(go.Bar(x=lis_x_max, y=lis_y_max, name="Max", marker=dict(color="LightSalmon")))
    fig.add_trace(go.Bar(x=lis_x, y=lis_y, name="Prob", marker=dict(color="CornflowerBlue")))
    return html.Div([dcc.Graph(figure=fig)], className="pb-3")
