import util
from models import DenseNet
import torch
from torch import nn
import numpy as np

from util import DirectedGraph

LAYERS_NEURONS=[128,64,16]
LEARNING_RATE=0.01

N_ITER=200000#NUMBER OF ITERATIONS/GENERATIONS
N_SESSIONS_PER_ITER=200
TRAIN_PERCENTILE=7#top percentile we are learning from each iteration
SUPER_PERCENTILE=6#top percentile of sesstions that live to the next generation

def play_game(model, graph, cur_state, action_prob)

def initilize_model_state(graph : DirectedGraph):
    init_state=[0]*(graph.num_vertices*3)

    #we start from node 0
    init_state[0]=1

    #set values to 1 at indices of nodes that are reachable from current node
    reachable_nodes=graph.edge_list[0]
    for i in range(graph.edge_cnt[0]):
        reachable_node=reachable_nodes[i]
        init_state[reachable_node]=1

    #set values of visited nodes so far to 1
    init_state[0]=1#not needed but for explanatory purposes this code is here

    return np.array(init_state)


def generate_session(model, graph):
    state_action_seq,reward=[],0
    softmax_fn=nn.Softmax(dim=0)

    cur_state=initilize_model_state(graph)
    while True:
        action_prob=softmax_fn(model(cur_state))

        action, transition_reward, next_state, terminal=play_game(model, graph, cur_state, action_prob)
        state_action_seq.append((cur_state, action))

        if terminal:
            reward=transition_reward
            break

        cur_state=next_state

    return (state_action_seq, reward)

def generate_sessions(model, graph, n_sessions):
    """
    Session consists of list that looks like: {[(s1,a1),(s2,a2),...,(sn,an)],r}, si=state in step i, ai=action in step i
    Code generates list of such sessions
    """
    sessions=np.empty((n_sessions, ), dtype=object)
    sessions.fill([])
    for i in range(n_sessions):
        sessions[i].append(generate_session(model))

    return sessions

def obtain_graph_walker_policy(graph):
    N=graph.num_vertices()

    #input to agent (neural network) consists of 3 parts each of length N
    #first part has 1 on index of next vertex we plan to move to else 0
    #second part has 1 on index of current vertex else 0
    #third part has 1 on index of (so far) visited vertex else 0
    word_length=3*N
    #output of agent as probability distribution over which state to choose next to go to
    #need to be careful because it might be that agent tries to go to next state that is not
    #offered to go to from current state
    action_space_length=N

    model = DenseNet([word_length, *LAYERS_NEURONS, action_space_length])
    optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for i in range(N_ITER):
        sessions = generate_sessions(model, graph, N_SESSIONS_PER_ITER)


if __name__=='__main__':
    obtain_graph_walker_policy(util.build_graph())

