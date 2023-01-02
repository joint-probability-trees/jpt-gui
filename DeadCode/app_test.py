import jpt
import igraph
from igraph import Graph, EdgeSeq

import plotly.graph_objects as go

import dash_bootstrap_components as dbc

from jpt.base.utils import list2interval

import dash
from dash import dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import math

global model
model: jpt.trees.JPT = jpt.JPT.load('test.datei')

global query_dict
query_dict = dict()

global evidence_dict
evidence_dict = dict()
# # ---TreeBuild&Draw---
# def tree_build(jpt_tree=None):
#     if jpt_tree is None:
#         return ([], [], [], [])
#     nodes = jpt_tree.root.recursive_children()
#     g: Graph = Graph()
#     g.add_vertices(len(nodes) + 1)
#     for n in nodes:
#         g.add_edge(n.parent.idx, n.idx)
#     lay = g.layout_reingold_tilford(mode="in", root= [0])
#     position = {k: lay[k] for k in range((len(nodes) + 1))}
#     Y = [lay[k][1] for k in range((len(nodes) + 1))]
#     M = max(Y)
#
#     es = EdgeSeq(g)  # sequence of edges
#     E = [e.tuple for e in g.es]  # list of edges
#
#     L = len(position)
#     Xn = [position[k][0] for k in range(L)]
#     Yn = [2 * M - position[k][1] for k in range(L)]
#     Xe = []
#     Ye = []
#     for edge in E:
#         Xe += [position[edge[0]][0], position[edge[1]][0], None]
#         Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]
#     return (Xe, Ye, Xn, Yn), list(range((len(nodes) + 1)))
#
#
# def draw_fig(koords=None, labels=None):
#     if labels is None:
#         labels = []
#     if koords is None:
#         koords = ([], [], [], [])
#
#     Xe = koords[0]
#     Ye = koords[1]
#     Xn = koords[2]
#     Yn = koords[3]
#
#     # Edges Draw
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=Xe,
#                              y=Ye,
#                              mode='lines',
#                              line=dict(color='rgb(210,210,210)', width=1),
#                              hoverinfo='none'
#                              ))
#     # Nodes Draw
#     fig.add_trace(go.Scatter(x=Xn,
#                              y=Yn,
#                              mode='markers',
#                              name='bla',
#                              marker=dict(symbol='circle-dot',
#                                          size=18,
#                                          color='#6175c1',  # '#DB4551',
#                                          line=dict(color='rgb(50,50,50)', width=1)
#                                          ),
#                              text=labels,
#                              hoverinfo='text',
#                              opacity=0.8
#                              ))
#     return fig
#

# --Modul_HTML--



modal_symbolic_q = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('Bitte wählen Sie die Werte aus')),
        dbc.ModalBody(
            [
                dcc.Dropdown(['0', '22', 'BANANA'], multi=True, id="symbolic_dd_q"),
            ]
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Abort", id="symbolic_abort_q", className="ms-auto", n_clicks=0),
                dbc.Button("Akzept", id="symbolic_akzept_q", className="ms-auto", n_clicks=0),
            ]
        )
    ],
    id="symbolic_query", is_open=False,
)

modal_numeric_q = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('Bitte wählen Sie die Werte aus')),
        dbc.ModalBody(
            [
                dcc.RangeSlider(0, 30, id="numeric_slider_q", value=[0, 30],allowCross=False, tooltip={"placement": "bottom", "always_visible": True})
            ]
        ),
        dbc.ModalFooter(
            [
            dbc.Button("Abort", id="numeric_abort_q", className="ms-auto", n_clicks=0),
            dbc.Button("Akzept", id="numeric_akzept_q", className="ms-auto", n_clicks=0),
            ]
        )
    ],
    id="numeric_query", is_open=False,
)


modal_query = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('Query')),
        dbc.ModalBody(
            [
                    dcc.Graph(id='query_tabel', figure=go.Figure(data=[go.Table(header=dict(values=['Variabel', 'Parameter']),
                                                                                cells=dict(values=[list(query_dict.keys()), list(query_dict.values())]))])),
                    html.Br(),
                    dcc.Dropdown(sorted(model.varnames), id='query_var'),
                    #dbc.Button("Symbolic", id="query_symbolic", className="ms-auto", n_clicks=0),
                    #dbc.Button("Numeric", id="query_numeric", className="ms-auto", n_clicks=0)
                    html.Br(),
                    dbc.Button("GO",id='query_go', className="ms-auto", n_clicks=0),
                    html.Div(id='query_res'),
                    ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="query_close", className="ms-auto", n_clicks=0)
        ),
        modal_numeric_q,
        modal_symbolic_q,
    ],
    id="modal_query", is_open=False,
)
#---------------------EVEDINCE MODAL-------------------------
modal_symbolic_e = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('Bitte wählen Sie die Werte aus')),
        dbc.ModalBody(
            [
                dcc.Dropdown(['0', '22', 'BANANA'], multi=True, id="symbolic_dd_e"),
            ]
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Abort", id="symbolic_abort_e", className="ms-auto", n_clicks=0),
                dbc.Button("Akzept", id="symbolic_akzept_e", className="ms-auto", n_clicks=0),
            ]
        )
    ],
    id="symbolic_evid", is_open=False,
)

modal_numeric_e = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('Bitte wählen Sie die Werte aus')),
        dbc.ModalBody(
            [
                dcc.RangeSlider(0, 30, id="numeric_slider_e", value=[0, 30],allowCross=False, tooltip={"placement": "bottom", "always_visible": True})
            ]
        ),
        dbc.ModalFooter(
            [
            dbc.Button("Abort", id="numeric_abort_e", className="ms-auto", n_clicks=0),
            dbc.Button("Akzept", id="numeric_akzept_e", className="ms-auto", n_clicks=0),
            ]
        )
    ],
    id="numeric_evid", is_open=False,
)


modal_evid = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle('Eivdenc')),
        dbc.ModalBody(
            [
                    dcc.Graph(id='evid_tabel', figure=go.Figure(data=[go.Table(header=dict(values=['Variabel', 'Parameter']),
                                                                                cells=dict(values=[list(evidence_dict.keys()), list(evidence_dict.values())]))])),
                    html.Br(),
                    dcc.Dropdown(sorted(model.varnames), id='evid_var'),
                    #dbc.Button("Symbolic", id="query_symbolic", className="ms-auto", n_clicks=0),
                    #dbc.Button("Numeric", id="query_numeric", className="ms-auto", n_clicks=0)
                    html.Br(),
                    dbc.Button("GO",id='evid_go', className="ms-auto", n_clicks=0),
                    html.Div(id='evid_res'),
                    ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="evid_close", className="ms-auto", n_clicks=0)
        ),
        modal_numeric_e,
        modal_symbolic_e,
    ],
    id="modal_evid", is_open=False,
)

#-----------------------------------------------------------





# Dash Start und Html Strucktur
app = dash.Dash('jpt Tree', external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks=True)
app.layout = html.Div(
    [
        # # Die BaumGraphic
        # dcc.Graph(id='treeFig', figure=go.Figure()),
        # # Input Schaltfläche um die Anzhal an Nodes zu Einzustellen
        # dcc.Input(id='Nodes',
        #           type='number',
        #           value=12),
        # dcc.Input(id='Zoom',
        #           type='number',
        #           value=0),

        dbc.Button("Query", id='query_open', n_clicks=0),
        modal_query,
        modal_evid,
        dcc.Store(id='tree'),
        dcc.Store(id='query')
    ])


# ---Baum Figure Funktion---
# Speichert die Eckdaten des Baumes ab in Quatrupel (Xe,Ye,Xn,Yn)
# tree = tree_build()


# # Liest Nodes Userinput von Dash, Erstellt einen Neuen Baum mit gewünscht viele Knoten
# def newTree(n, debug=True):
#     # IO Abfangen!
#     with open('test.datei') as f:
#         try:
#             tree = f.readlines()
#         except ModuleNotFoundError:
#             raise Exception('Datei wurd net gefunden')
#     if debug:
#         j_sampel: jpt.trees.JPT = jpt.JPT.load('test.datei')
#     else:
#         j_sampel: jpt.trees.JPT = jpt.JPT.load('test2.datei')
#     tree, label = tree_build(j_sampel)
#     return draw_fig(tree, label), tree
#
#
# # Router der Alle Inptus verarbeiten Kann die Tree bearbeiten wollen Links:
# # https://gist.github.com/nicolaskruchten/8aa6ae9df62d2c45ef87ad28efd06a31
# # https://github.com/plotly/dash-docs/issues/961
#
# @app.callback(
#     Output('treeFig', 'figure'),
#     Output('tree', 'data'),
#     Input('Nodes', 'value'),
#     Input('Zoom', 'value'),
#     Input('tree', 'data'))
# def tree_router(nodes: int, zoom: int, tree_data):
#     cb = ctx.triggered_id
#     print(cb, nodes, zoom)
#     if cb == "Nodes":
#         print("node")
#         return newTree(nodes)
#     elif cb == "Zoom":
#         print("zoom")
#         return newTree(nodes, False)  # zooming(zoom)
#     else:
#         return draw_fig(), tree_data
#         # return draw_fig(tree, list(map(str, range(1, len(tree[3]) + 1))))
#

# ---MODAL Function---

@app.callback(
    Output('modal_query', 'is_open'),
    [
        Input('query_open', 'n_clicks'),
        Input('query_close', 'n_clicks'),
     ],
    [State('modal_query', 'is_open')]
)
def toggel_modal_query(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('symbolic_query', 'is_open'),
    Output('numeric_query', 'is_open'),
    Output('numeric_slider_q', "min"),
    Output('numeric_slider_q', "max"),
    Output('symbolic_dd_q', 'options'),
    Output('query_tabel', 'figure'),
    Input('query_var', 'value'),
    [
        Input('symbolic_abort_q', 'n_clicks'),
        Input('symbolic_akzept_q', 'n_clicks'),
        Input('numeric_abort_q', 'n_clicks'),
        Input('numeric_akzept_q', 'n_clicks')
    ],
    Input('numeric_slider_q', 'value'),
    Input('symbolic_dd_q', 'value')
)
def query_value_selector(variable_name, n1, n2, n3, n4, num_val, sym_val):
    #figure = go.Figure(data=[go.Table(header=dict(values=['Variabel', 'Parameter']),
    #       cells=dict(values=[list(query_dict.keys()), list(query_dict.values())]))])
    figure = go.Figure(data=[go.Table(header=dict(values=['Variabel', 'Parameter']),
                                      cells=dict(values=[list(query_dict.keys()), list(query_dict.values())]))])
    print("Teees", variable_name)
    if not variable_name:
        print('not')
        return False, False, 0, 100, [], figure
    variable = model.varnames[variable_name]
    cb = ctx.triggered_id
    if cb == 'symbolic_akzept_q' or cb == 'numeric_akzept_q' or cb == 'symbolic_abort_q' or cb == 'numeric_abort_q':
        if cb == 'symbolic_akzept_q':
            query_dict.update({variable_name: sym_val})
        if cb == 'numeric_akzept_q':
            query_dict.update({variable_name: num_val})

        figure = go.Figure(data=[go.Table(header=dict(values=['Variabel', 'Parameter']),
                                          cells=dict(values=[list(query_dict.keys()), list(query_dict.values())]))])
        return False, False, 0, 100, [], figure

    if variable.symbolic:
        print('sym')
        return True, False, 0, 10, list(variable.domain.labels.keys()), figure

    elif variable.numeric:
        print('num')
        expectation = model.expectation([variable], {}, confidence_level=1.)
        print(expectation[variable])
        if math.isnan(expectation[variable].lower):r(n_clicks,):
            return False, True, math.ceil(expectation[variable].upper), math.ceil(expectation[variable].upper)+1, [], figure
        elif math.isnan(expectation[variable].upper):
            return False, True, math.ceil(expectation[variable].lower), math.ceil(expectation[variable].lower)+1, [], figure
        else:
            return False, True, math.floor(expectation[variable].lower), math.ceil(expectation[variable].upper), [], figure


#-----------------------------EVID FUNCTION----------------------
# @app.callback(
#     Output('modal_evid', 'is_open'),
#     [
#         Input('evid_open', 'n_clicks'),
#         Input('evid_close', 'n_clicks'),
#      ],
#     [State('modal_evid', 'is_open')]
# )
# def toggel_modal_query(n1, n2, is_open):
#     if n1 or n2:
#         return not is_open
#     return is_open

@app.callback(
    Output('symbolic_evid', 'is_open'),
    Output('numeric_evid', 'is_open'),
    Output('numeric_slider_e', "min"),
    Output('numeric_slider_e', "max"),
    Output('symbolic_dd_e', 'options'),
    Output('evid_tabel', 'figure'),
    Input('evid_var', 'value'),
    [
        Input('symbolic_abort_e', 'n_clicks'),
        Input('symbolic_akzept_e', 'n_clicks'),
        Input('numeric_abort_e', 'n_clicks'),
        Input('numeric_akzept_e', 'n_clicks')
    ],
    Input('numeric_slider_e', 'value'),
    Input('symbolic_dd_e', 'value')
)
def evid_value_selector(variable_name, n1, n2, n3, n4, num_val, sym_val):
    #figure = go.Figure(data=[go.Table(header=dict(values=['Variabel', 'Parameter']),
    #       cells=dict(values=[list(query_dict.keys()), list(query_dict.values())]))])
    figure = go.Figure(data=[go.Table(header=dict(values=['Variabel', 'Parameter']),
                                      cells=dict(values=[list(evidence_dict.keys()), list(evidence_dict.values())]))])
    print("Teees", variable_name)
    if not variable_name:
        print('not')
        return False, False, 0, 100, [], figure
    variable = model.varnames[variable_name]
    cb = ctx.triggered_id
    if cb == 'symbolic_akzept_e' or cb == 'numeric_akzept_e' or cb == 'symbolic_abort_e' or cb == 'numeric_abort_e':
        if cb == 'symbolic_akzept_e':
            evidence_dict.update({variable_name: set(sym_val)})
        if cb == 'numeric_akzept_e':
            evidence_dict.update({variable_name: [float(v) for v in num_val]})

        figure = go.Figure(data=[go.Table(header=dict(values=['Variabel', 'Parameter']),
                                          cells=dict(values=[list(evidence_dict.keys()), list(evidence_dict.values())]))])
        return False, False, 0, 100, [], figure

    if variable.symbolic:
        print('sym')
        return True, False, 0, 10, list(variable.domain.labels.keys()), figure

    elif variable.numeric:
        print('num')
        expectation = model.expectation([variable], {}, confidence_level=1.)
        print(expectation[variable])
        if math.isnan(expectation[variable].lower):
            return False, True, math.ceil(expectation[variable].upper), math.ceil(expectation[variable].upper)+1, [], figure
        elif math.isnan(expectation[variable].upper):
            return False, True, math.ceil(expectation[variable].lower), math.ceil(expectation[variable].lower)+1, [], figure
        else:
            return False, True, math.floor(expectation[variable].lower), math.ceil(expectation[variable].upper), [], figure



#-----------------------------------------------------------------






@app.callback(
    Output('modal_evid', 'is_open'),
    [
        #Input('evid_open', 'n_clicks'),
        Input('evid_close', 'n_clicks'),
        Input('query_go', 'n_clicks'),
     ],
    [State('modal_evid', 'is_open')],

)
def evid_toggler(n1, n2, is_open):

    if n1 or n2:
        return not is_open
    return is_open
@app.callback(
    Output('evid_res', 'children'),
    Input('evid_go', 'n_clicks'),
)
def infer(n_clicks,):
    print('-------------------------------------')
    if n_clicks <= 0:
        return "Oopsi Poopsi"

    evidence = jpt.variables.VariableMap([(model.varnames[k], v) for k, v in evidence_dict.items()])
    query = jpt.variables.VariableMap([(model.varnames[k], v) for k, v in query_dict.items()])

    result = model.infer(query, evidence)

    return result.format_result()

# @app.callback(
#     Output('symbolic_query', 'is_open'),
#     [
#         Input('query_symbolic', 'n_clicks'),
#         Input('symbolic_abort', 'n_clicks'),
#         Input('symbolic_akzept', 'n_clicks'),
#     ],
#     State('symbolic_query', 'is_open')
# )
# def toggel_sym_modal(n1,n2,n3, is_open):
#     if n1 or n2 or n3:
#         return not is_open
#     return is_open
#
# @app.callback(
#     Output('numeric_query', 'is_open'),
#     [
#         Input('query_numeric', 'n_clicks'),
#         Input('numeric_abort', 'n_clicks'),
#         Input('numeric_akzept', 'n_clicks'),
#     ],
#     State('numeric_query', 'is_open')
# )
# def toggel_sym_modal(n1,n2,n3, is_open):
#     if n1 or n2 or n3:
#         return not is_open
#     return is_open

if __name__ == '__main__':
    # tree = tree_build(j_sampel)
    app.run_server(debug=True, )






'''TODOO
States für den Derzeitigen Tree
Tree String in GUI
Zoom Tree eifnach anstast Root andrenn Node Build Tree ändern
Leafs Grün Ferben allnodes hat 2 Cahnmaps Keys sind IDs wann welche Farbe'''
