import dash
from dash import html, Input, Output, callback
from src import components as c

dash.register_page(__name__, path='/')


@callback(
    Output('list', 'children'),
    Input('list', 'children')
)
def gen_varnames(children):
    var_divs = []
    variabels = list(c.in_use_tree.varnames)
    if (len(variabels) <= 1):
        return var_divs
    for var_name in variabels:

        variable = c.in_use_tree.varnames[var_name]
        if variable.numeric:
            c.priors[variable.name]
            mini = c.priors[variable.name].cdf.intervals[0].upper
            maxi = c.priors[variable.name].cdf.intervals[-1].lower
            stri = f" {var_name} ∈ [{round(mini,3)}, {round(maxi, 3)}]"
            var_divs.append(html.Div(stri))
        else:
            print(list(c.priors[variable.name].labels.keys()))
            vals = list(variable.domain.labels.values())
            #c.priors[var_name]
            stri = f"{var_name} ∈ ({vals})"
            var_divs.append(html.Div(stri))
    return var_divs

    return html.Div(children=var_divs)


layout = html.Div([
    html.H1("Home"),
    html.Div(children=[], id="list"),
])

# Home Name , Type , Range, AUGEN EMOTION HTML Sektion bigger and better