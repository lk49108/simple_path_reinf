import util
import torch
from torch import nn
import numpy as np
import time

from tqdm import trange
from numba import njit,jit,objmode,prange

import multiprocessing
import concurrent

import functools

from models import DenseNet
from train import train_network
from util import DirectedGraph



LAYERS_NEURONS=[128,64,16,8]
LEARNING_RATE=0.0001

N_ITER=200#NUMBER OF ITERATIONS/GENERATIONS
N_SESSIONS_PER_ITER=5000
TRAIN_PERCENTILE=10#top percentile we are learning from each iteration
SUPER_PERCENTILE=1#top percentile of sesstions that live to the next generation

epsilon=0.05#probability to select random choice when moving along edges of a graph

def make_choice(model, graph : DirectedGraph, cur_state):
    reachable_nodes_one_hot=util.extract_legal_actions(cur_state)
    visited_nodes_one_hot=util.extract_visited_nodes(cur_state)

    valid_actions_one_hot=reachable_nodes_one_hot & (~visited_nodes_one_hot)
    valid_actions=np.where(valid_actions_one_hot)[0]

    if len(valid_actions) <= 0:
        #terminate the episode
        return -1, 0, True

    if np.random.uniform()<=epsilon:
        #select a random choice
        action = np.random.choice(valid_actions)
    else:
        action_prob : np.ndarray = nn.Softmax(dim=1)(model(cur_state[None, :]))[0,:].detach().numpy()  # make input be 2D before giving it to model and then reverse

        valid_actions_prob=action_prob[valid_actions_one_hot]
        valid_actions_prob+=1e-9#to solve numerical stability problems when sum(valid_actions_prob) is close to 0
        valid_actions_prob/=np.sum(valid_actions_prob)

        action=np.random.choice(valid_actions, p=valid_actions_prob)

    return action, 1, False

def choose_start_node(graph):
    #dummy choosing of starting node by choosing uniformly random between N of them
    return np.random.choice(graph.N)

def initilize_model_state(graph : DirectedGraph):
    init_state=[0]*(graph.N*3)

    init_node=choose_start_node(graph)

    #set 1 to index of current node (we start from node 0)
    offset=0
    init_state[offset+init_node]=1

    #set values to 1 at indices of nodes that are reachable from current node
    offset=graph.N
    reachable_nodes=graph.edge_list[init_node]
    for i in range(graph.edge_cnt[init_node]):
        reachable_node=reachable_nodes[i]
        init_state[offset+reachable_node]=1

    #set values of visited nodes so far to 1
    offset=graph.N*2
    init_state[offset+init_node]=1#not needed but for explanatory purposes this code is here

    return torch.from_numpy(np.array(init_state)).float()

def update_model_state(cur_state, action, graph : DirectedGraph):
    next_state=[0]*(graph.N*3)

    #set 1 to index of current node
    offset=0
    next_state[offset+action]=1

    #set 1 to indices of nodes that are reachable from current node
    offset=graph.N
    reachable_nodes=graph.edge_list[action]
    for i in range(graph.edge_cnt[action]):
        reachable_node=reachable_nodes[i]
        next_state[offset+reachable_node]=1


    #set values of visited nodes so far to 1
    offset=graph.N*2
    next_state[offset:]=cur_state[offset:]
    next_state[offset+action]=1

    return torch.from_numpy(np.array(next_state)).float()


def generate_session(model, graph):
    state_action_seq,reward=[],0

    cur_state=initilize_model_state(graph)
    while True:
        action, transition_reward, terminal=make_choice(model, graph, cur_state)

        if terminal:
            break

        reward+=transition_reward
        state_action_seq.append((cur_state.detach().numpy(), action))

        cur_state=update_model_state(cur_state, action, graph)

    return (state_action_seq, reward)

def generate_sess(model, graph, i):
    """
    Help function to allow multiprocessing.
    """
    return generate_session(model, graph)

def generate_sessions(model, graph, n_sessions):
    """
    Session consists of list that looks like: {[(s1,a1),(s2,a2),...,(sn,an)],r}, si=state in step i, ai=action in step i
    Code generates list of such sessions
    """

    gen_sess_fcn = functools.partial(generate_sess, model, graph)

    with multiprocessing.Pool(processes=int(multiprocessing.cpu_count())) as pool:
        sessions = pool.map(gen_sess_fcn, range(n_sessions))
    # with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:#15.2325279712677
    #     pool.map(gen_sess_fcn, range(n_sessions))

    # for i in range(n_sessions):#11.512
    #     sessions[i]=generate_session(model, graph)

    return sessions

def select_top_sessions(sessions, percentile):
    N=len(sessions)
    top_num=int(1+N*percentile/100)

    sessions.sort(key=lambda x : x[1], reverse=True)
    return sessions[:top_num]

def select_super_sessions(sessions):
    return select_top_sessions(sessions, SUPER_PERCENTILE)


def select_elite_sessions(sessions):
    return select_top_sessions(sessions, TRAIN_PERCENTILE)

def create_data_loader(N, sessions):
    x, y = np.empty((0, 3*N)), np.empty((0, 1))
    for state_action_pairs, _ in sessions:
        for state, action in state_action_pairs:
            x = np.append(x, state[None, :], axis=0)
            y = np.append(y, np.array([[action]]), axis=0)

    dataset = torch.from_numpy(np.column_stack((x, y))).float()
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=16)

    return loader

def obtain_graph_walker_policy(graph):
    #input to agent (neural network) consists of 3 parts each of length N
    #first part has 1 on index of next vertex we plan to move to else 0
    #second part has 1 on index of current vertex else 0
    #third part has 1 on index of (so far) visited vertex else 0
    word_length=3*graph.N
    #output of agent as probability distribution over which state to choose next to go to
    #need to be careful because it might be that agent tries to go to next state that is not
    #offered to go to from current state
    action_space_length=graph.N

    model = DenseNet([word_length, *LAYERS_NEURONS, action_space_length])
    optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    super_sessions = None

    mean_reward_vals=[]
    for i in range(N_ITER):
        t=time.time()
        sessions : list(tuple(list(tuple), int)) = generate_sessions(model, graph, N_SESSIONS_PER_ITER)
        sessions_gen_time=time.time()-t
        print('Sessions generation time: {}'.format(sessions_gen_time))

        t=time.time()
        if i>0:
            #append super sessions (top SUPER_PERCENTILE from previous iteration) to current sessions
            sessions+=super_sessions
        super_sessions_append_time=time.time()-t
        print('Supper sessions append time: {}'.format(super_sessions_append_time))

        t=time.time()
        elite_sessions=select_elite_sessions(sessions)#select sessions on which we train the agent
        super_sessions=select_super_sessions(sessions)#select sessions which survive to next iteration
        elite_and_super_sessions_select_time=time.time()-t
        print('Elite and supper sessions select time: {}'.format(elite_and_super_sessions_select_time))

        t=time.time()
        train_loader=create_data_loader(graph.N, elite_sessions)
        train_loader_creation_time=time.time()-t
        print('Train loader creation time: {}'.format(train_loader_creation_time))

        t=time.time()
        train_network(model, torch.nn.CrossEntropyLoss(), optimizer, train_loader)
        train_time=time.time()-t
        print('Train time: {}'.format(train_time))

        reward_super_sessions_mean=np.mean([el[1] for el in super_sessions])
        print('Mean reward of top {} % sessions = {}'.format(SUPER_PERCENTILE, reward_super_sessions_mean))

        if i>=5:
            best_session_state_action_pairs, best_session_reward = super_sessions[0]
            print('Best session reward: {}'.format(best_session_reward))
            print(918, end='')
            for state, action in best_session_state_action_pairs:
                print('->{}'.format(action), end='')

            print()
            break

        if i>20 and sum([abs(reward_super_sessions_mean-mean_reward_vals[j]) for j in range(-1,-5,-1)])<0.1:
            best_session_state_action_pairs,best_session_reward=super_sessions[0]
            print('Best session reward: {}'.format(best_session_reward))
            print(918,end='')
            for state, action in best_session_state_action_pairs:
                print('->{}'.format(action),end='')

            print()
            break

        mean_reward_vals.append(reward_super_sessions_mean)


if __name__=='__main__':
    obtain_graph_walker_policy(util.build_graph3(n_chains=20,chains_length=10))