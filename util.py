import numpy as np

class DirectedGraph:

    def __init__(self, N):
        self.N=N
        self.adj_matr=np.zeros((N, N), dtype=np.int8)
        self.edge_list=np.full((N, N), None)
        self.edge_cnt=np.zeros(N, dtype=np.int16)

    def add_edge(self, v1,v2):
        if self.adj_matr[v1][v2]==1:
            return

        self.adj_matr[v1][v2]=1
        self.edge_list[v1][self.edge_cnt[v1]]=v2
        self.edge_cnt[v1]+=1

    def __repr__(self):
        str='N: {}'.format(self.N)
        str+=', edges: '
        for v1 in range(self.N):
            for v2 in self.edge_list[v1]:
                if v2 is None:
                    break

                str+='{}->{}, '.format(v1,v2)

        str=str[:-2]

        return str

def build_graph():
    graph=DirectedGraph(5)
    edge_list=[(0,1),(3,2),(2,1),(4,1),(1,3),(2,3)]
    for edge in edge_list:
        graph.add_edge(*edge)

    return graph

def build_graph2():
    np.random.seed(73)

    N=50
    rand_perm=np.random.permutation(N)
    print(rand_perm)

    graph=DirectedGraph(N)
    for i in range(N-1):
        graph.add_edge(rand_perm[i], rand_perm[i+1])

    for i in range(N*(N-1)//5):
        graph.add_edge(np.random.randint(0, N), np.random.randint(0, N))

    return graph

def extract_legal_actions(state : list):
    N=len(state)//3

    filter=state[N:2*N]
    return set(np.where(filter==1)[0])


def extract_visited_nodes(state : list):
    N=len(state)//3

    filter=np.array(state[2*N:])
    return set(np.where(filter==1)[0])

if __name__=='__main__':
    graph=build_graph2()
    print(graph)

