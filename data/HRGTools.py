import random

def generate_random_graph(n,e):
    """
    Generate random graph with number of n
    :param n: graph's number of nodes
    :return: a dict of graph
    """
    number = n
    avg_edges=e
    node_list = []
    graph = {}
    for node in range(number):
        node_list.append(node)


    for node in node_list:
        graph[node] = []

    for node in node_list:
        # gauss, avg, square
        edges = random.gauss(avg_edges, 10)
        edges = int(edges)
        for i in range(edges):
            index = random.randint(0, number - 1)
            node_append = node_list[index]
            if node_append not in graph[node] and node != node_append:
                graph[node].append(node_append)
                graph[node_append].append(node)
    return graph


def generate_edge(graph):
    """
    draw the edge of graph
    :param graph: a dict of graph
    :return: a list of edge
    """
    edges = []
    for node in graph:
        for neighbour in graph[node]:
            edges.append((node, neighbour))
    return edges


if __name__ == "__main__":

    nodes = 75
    edges = 20
    graph = generate_random_graph(nodes, edges)
    lists = generate_edge(graph)
    f = open("path to a file to store the graph", "w+")
    for i in lists:
        f.write(str(i[0])+"\t"+str(i[1])+"\r\n")
    f.close()

