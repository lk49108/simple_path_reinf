import numpy as np

class DirectedGraph:

    def __init__(self, N):
        self.N=N
        self.adj_matr=np.zeros((N, N), dtype=np.int8)
        self.edge_list=np.full((N, N), None)
        self.edge_cnt=np.zeros(N, dtype=np.int16)
        self.tot_edge_cnt=0

    def add_edge(self, v1,v2):
        if self.adj_matr[v1][v2]==1:
            return

        self.adj_matr[v1][v2]=1
        self.edge_list[v1][self.edge_cnt[v1]]=v2
        self.edge_cnt[v1]+=1

        self.tot_edge_cnt+=1

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

    for i in range(N*(N-1)//100):
        graph.add_edge(np.random.randint(0, N), np.random.randint(0, N))

    return graph

def build_graph3(n_chains=10, chains_length=10):
    def build_chain(graph, node_ids):
        for i in range(len(node_ids)-1):
            graph.add_edge(node_ids[i], node_ids[i+1])

        return node_ids[0]

    def add_random_edges(graph, subgraph_node_ids):
        N=len(subgraph_node_ids)
        for i in range(len(subgraph_node_ids)):
            graph.add_edge(subgraph_node_ids[np.random.randint(0, N)], subgraph_node_ids[np.random.randint(0, N)])


    np.random.seed(73)

    N=n_chains*chains_length+2
    rand_perm=np.random.permutation(N)

    graph=DirectedGraph(N)
    start_node=rand_perm[-1]
    for i in range(n_chains):
        chain_start_node=build_chain(graph, rand_perm[i*chains_length:(i+1)*chains_length])
        graph.add_edge(start_node, chain_start_node)

    graph.add_edge(rand_perm[-3], rand_perm[-2])

    for i in range(n_chains):
        if i<n_chains-1:
            add_random_edges(graph, rand_perm[i*chains_length:(i+1)*chains_length])
        else:
            add_random_edges(graph, rand_perm[i*chains_length:(i+1)*chains_length+1])

    print('Longest simple path: ',end='')
    print(rand_perm[-1],end='')
    for i in range(-chains_length-2, -1, 1):
        print('->{}'.format(rand_perm[i]),end='')
    print()

    return graph


def extract_legal_actions(state : list):
    N=len(state)//3

    filter=np.array(state[N:2*N])
    return filter==1


def extract_visited_nodes(state : list):
    N=len(state)//3

    filter=np.array(state[2*N:])
    return filter==1

if __name__=='__main__':
    graph=build_graph3(2,2)
    print(graph)
    print(graph.tot_edge_cnt)
