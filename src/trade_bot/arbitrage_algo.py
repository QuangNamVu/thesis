from dijkstar import Graph, find_path


graph = Graph()

graph.add_edge('A', 'B', {'cost': -1})
graph.add_edge('B', 'C', {'cost': -2})
graph.add_edge('C', 'A', {'cost': -3})
cost_func = lambda u, v, e, prev_e: e['cost']
find_path(graph, 1, 2, cost_func=cost_func)