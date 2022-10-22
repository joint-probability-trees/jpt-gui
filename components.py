import math
from dash import dcc, html
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


def div_to_variablemap(model: jpt.trees.JPT, variables: list, constrains: list) -> jpt.variables.VariableMap:
    """
    Transforms variable and Constrains List form the GUI to a VariableMap
    :param model: the JPT model of the Prob. Tree
    :param variables: The list of chosen Variables
    :param constrains:  The list of for the Variables on the same Index
    :return: VariableMap of the Variables with its associated Constraints
    """
    var_dict = {}
    print(f'vars:{variables}  , cons{constrains}')
    for i in range(0, len(variables) - 1):
        if variables[i] is None: #TODOO WIESO HAT DAS NULLS
            break
        variable = model.varnames[variables[i]]
        if variable.numeric:
            var_dict.update({variables[i]: constrains[i]})
        else:
            var_dict.update({variables[i]: set(constrains[i])})
        return jpt.variables.VariableMap([(model.varnames[k], v) for k, v in var_dict.items()])


def mpe_result_to_div(model: jpt.trees.JPT, res: list[jpt.trees.MPEResult]) -> list:
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
                     options={k: v for k, v in zip(variable.domain.labels.keys(), variable.domain.labels.values())},
                     value=list(restriction), multi=True, disabled=True, className="ps-3")],
                style={"display": "grid", "grid-template-columns": "30% 70%"})]
        return_div += [html.Div(className="pt-1")]

    return_div = [html.Div([dcc.Dropdown(options=["Likelihood"], value="Likelihood", disabled=True,
                                          className="margin10"),
                             dcc.Dropdown(options=[res.result],value=res.result, disabled=True, className="ps-3 pb-2")],
                            id="likelihood", style={"display": "grid", "grid-template-columns": "30% 70%"})] + return_div

    return return_div


def create_prefix_text_query(len_fac_q, len_fac_e) -> list:
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
                        'padding-top': (len_fac_q * 1 if len_fac_q >= len_fac_e else len_fac_e) * 1}),
    ]


def create_prefix_text_mpe(len_fac):
    """
    Creates Dash Style Prefix for the MPE GUI
    :param len_fac: Length of Evidence input used for Scaling
    :return: Children div for the prefix MPE GUI
    """
    return [
        html.Div("argmax ", className="pe-3",
                 style={"width": "30%", 'fontSize': len_fac * 10 if len_fac * 10 < 30 else 30}),
        html.Div("P ", className="ps-3",
                 style={"width": "30%", "height": "100%", 'fontSize': len_fac * 15 if len_fac * 15 < 75 else 75}),
    ]


def generate_free_variables_from_div(model: jpt.trees.JPT, variable_div: list):
    variable_list = variable_div
    variables = []
    for v in variable_list:
        if len(v['props']) > 2:
            print(v)
            variables += [v['props']['value']]
    return generate_free_variables_from_list(model, variables)


def generate_free_variables_from_list(model: jpt.trees.JPT, variable_list):
    vars_free = model.varnames.copy()
    # return [v for v in model.varnames if v not in varaible_list]
    print(variable_list)
    for v in variable_list: vars_free.pop(v)
    return list(vars_free.keys())


def update_free_vars_in_div(model, variable_div):
    variable_list = variable_div
    vars_free = generate_free_variables_from_div(model, variable_list)

    for v in variable_list:
        if len(v['props']) > 2:
            print(v)
            v['props']['options'] = [v['props']['value']] + vars_free

    return variable_list


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

    variable_list.pop(del_index)
    constrains_list.pop(del_index)

    new_var_list = update_free_vars_in_div(model, variable_list)
    return new_var_list, constrains_list


def add_selector_to_div(model: jpt.trees.JPT, variable_div, constrains_div, type: str,
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
        dcc.Dropdown(id={'type': type, 'index': index},
                     options=variable_list[0]['props']['options'][1:]))
    constrains_list.append(dcc.Dropdown(id={'type': type, 'index': index}, disabled=True))
    return variable_list, constrains_list
