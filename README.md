# simple_path_reinf

Project to solve NP problem of finding longest simple path in directed graph G=(V,E) using reinforcement learning approach.
We denote by n=|V|.

Neural network will be used as a policy.
Input to the network will be current state of the agent
and on the output network will give distribution over actions
to take in current state.
State of the agent needs to consist of information
saying which states he already visited, current state he is in and next state we consider going to (the one that is adjacent to current one and is not yet visited). We might also give agent as part of state adjacency matrix which is of dimension n^2. But not for now since the dimension of the same is a big problem and might get agent completely lost.
Output of the agent needs to consist of probability weather
to accept transition from current state to next state.
