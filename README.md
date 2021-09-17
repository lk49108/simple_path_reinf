# simple_path_reinf

Project to solve NP hard problem of finding longest simple path in directed graph using reinforcement learning approach. At the moment Cross entropy algorithm is used. We will implement other reinforcement learning algorithms later.

Neural network is used as a policy. State (input to the network) is binary (consists of 0's and 1's) of length 3*N, where N is number of nodes in the graph. First N digits of state are one-hot encoding of current node agent is in. Second batch of N digits has one-hot encoding of nodes that are reachable from current node agent is in. Third batch of N digits is one-hot encoding of what nodes are visited so far (in current session) by agent. Output of the agent (network) is probability distribution over what node should agent go to next.

PS-some stuff are still hardcoded in the code and kind of strange because of testing purposes. Additionally, time logging and tqdm library is used to see what are performance bottlenecks in code and code will be optimized accordingly in order to be able to run the algorithm on huge (>10000 nodes) graphs.