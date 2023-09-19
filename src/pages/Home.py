import dash
from dash import html, Input, Output, callback
import components as c
import dash_bootstrap_components as dbc

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
            childStr = [html.Div(var_name, className="fs-4  flex-nowrap flex-grow-0 text-nowrap text-start"), html.Div(" ∈ ", className="pe-2 ps-1 fs-4  flex-nowrap flex-grow-0 text-nowrap text-start"), html.Div(f"[{round(mini,3)}, {round(maxi, 3)}]", className="fs-4  flex-nowrap flex-grow-0 text-nowrap text-start")]
            var_divs.append(html.Div(childStr, className="d-flex justify-content-center flex-grow-0"))
        else:
            vals = list(variable.domain.labels.values())
            #c.priors[var_name]
            stri = f"{var_name} ∈ ({vals})"
            childStr = [html.Div(var_name, className="fs-4 flex-nowrap flex-grow-0 text-nowrap text-start"), html.Div(" ∈ ", className="pe-2 ps-1 fs-4 flex-nowrap flex-grow-0 text-nowrap text-start"), html.Div(f"({vals})", className="fs-4 flex-nowrap flex-grow-0 text-nowrap text-start")]
            var_divs.append(html.Div(childStr, className="d-flex justify-content-center flex-grow-0"))
    return var_divs

    return html.Div(children=var_divs)


layout = html.Div([
    dbc.Row(html.H1("Home"), className="d-flex justify-content-center"),
    dbc.Row(dbc.Col(html.Img(src="./assets/Logo_JPT_White.png", height="200px"), width=3), className="d-flex justify-content-center mb-5 ms-5 ps-3"),
    dbc.Row(html.Div(children=[], id="list", className=""), className="d-flex justify-content-center"),
])

# Home Name , Type , Range, AUGEN EMOTION HTML Sektion bigger and better