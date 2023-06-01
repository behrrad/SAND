import json
import pickle
import networkx as nx
import tqdm
from networkx.readwrite import json_graph
import sqlite3


GET_LABEL_QUERY = '''select label from mapping where wid = ?'''


def connect_to_db():
    conn = sqlite3.connect('relation.db')
    return conn.cursor()


def get_parents_to_the_root(cursor, graph, wid):
    data = [wid]
    checked_parents = []
    while len(data) > 0:
        edges = graph.in_edges(data[0])
        for edge in list(edges):
            if list(edge)[0] in checked_parents:
                continue
            checked_parents.append(list(edge)[0])
        data = list(dict.fromkeys(data))
        labels = []
        for row in data:
            cursor.execute(GET_LABEL_QUERY, (row,))
            labels.append(cursor.fetchone())
        # print("parent:")
        # checked_parents.append(data[0])
        del data[0]
        del labels[0]
        # print(labels)
    return checked_parents


def main():
    with open('./typeHierarchy.pickle', 'rb') as f:
        cursor = connect_to_db()
        graph = pickle.load(f)
        # print(graph.number_of_edges())
        nodes = list(graph.nodes)
        # print(len(nodes))
        x = 0
    data = input()
    ids = get_parents_to_the_root(cursor, graph, data)
    labels = []
    for wid in tqdm.tqdm(ids):
        cursor.execute(GET_LABEL_QUERY, (wid, ))
        labels.append(cursor.fetchone())
    print(ids)
    print(labels)
    '''
    for node in tqdm.tqdm(nodes):
        cursor.execute(GET_LABEL_QUERY, (node, ))
        label = cursor.fetchone()
        edges = graph.in_edges(node)
        out_edges = graph.out_edges(node)
        labels = []
        # if len(list(edges)) > 0:
        #     continue
        for edge in list(edges):
            cursor.execute(GET_LABEL_QUERY, (list(edge)[0],))
            labels.append(cursor.fetchone())
        out_edges_labels = []
        for edge in list(out_edges):
            cursor.execute(GET_LABEL_QUERY, (list(edge)[1],))
            out_edges_labels.append(cursor.fetchone())
        print(label)
        print("in edges:")
        print(labels)
        print("out edges:")
        print(out_edges_labels)
        x += 1
        if x >= 10:
            break
    print(x)
    '''


if __name__ == '__main__':
    main()
