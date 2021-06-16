import numpy as np

class DirectedGraph:

    def __init__(self, N):
        self.N=N
        self.adj_matr=np.zeros((N,N))
        self.edge_list=np.full((N,N), None)
        self.edge_cnt=np.zeros(N)

    def add_edge(self, v1,v2):
        self.adj_matr[v1][v2]=1
        self.edge_list[v1][self.edge_cnt[v1]]=v2
        self.edge_cnt[v1]+=1



def build_graph():
    graph=DirectedGraph(5)
    edge_list=[(0,1),(3,2),(2,1),(4,1),(1,3),(2,3)]
    for edge in edge_list:
        graph.add_edge(*edge)

    return graph