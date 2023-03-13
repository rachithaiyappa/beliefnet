"""
This script performs the weighted belief network dynmaics on a star graph and 
tracks the evolution of the hub's belief systems in response to its neighbors,
a fraction of which are zealots. 
The belief system of the zealots are fixed throughout the simulation and are different from that of the hub's initial belief system

 
It returns a nested dictionary  which, an example of which is shown below
result_single_run[alpha][beta][zealot_fraction] = ['stable', 'stableplus', 'stableplus', ...]
zealot_fraction ranges from 0 to 1 where 0 (1) indicates that none (all) of the hub's neighbors are zealots.

Author: Rachith Aiyappa
"""


from beliefnet.model import star_graph_dynamics
import pickle as pkl

if __name__ == "__main__":

    alpha = 1.5 # weight of social influence
    beta = 1 # weight of internal coherence
    N = 40 # number of nodes in the population
    n = 3 # number of nodes in each individuals' belief system
    hub_kind = 'stable' # initial stability of the hub.

    result_single_run = star_graph_dynamics(alpha=alpha,
                                            beta=beta,
                                            N=N,
                                            n=n,
                                            hub_kind=hub_kind,
                                            )

    
    with open(f"ZealotFraction_vs_HubBelief_TimeEvolution--alpha{alpha}_beta{beta}_N{N}_hubkind{hub_kind}.pkl","wb") as f:
        pkl.dump(result_single_run,f)
    
