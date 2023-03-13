"""Belief network model."""
import itertools
import math
import random
from functools import reduce
from operator import mul
import networkx as nx
import numpy as np
from scipy.stats import norm


def gnp_belief_network(n_nodes: int, prob: float, seed: int) -> nx.Graph:
    """create an ER random belief network (initialized with random beliefs)."""
    np.random.seed(seed)
    G = nx.gnp_random_graph(n_nodes, prob, seed=seed)
    initialize_with_random_beliefs(G, seed)
    return G


def initialize_with_random_beliefs(G: nx.Graph, seed: int):
    """initialize a belief network by assigning random weights (-1, 1)."""
    nx.set_edge_attributes(
        G, {edge: np.random.random_sample() * 2.0 - 1.0 for edge in G.edges()}, "belief"
    )


def complete_belief_network(N: int, edge_values="default") -> nx.Graph:
    """create a belief network which is fully connected and with all edges set to a user input (1) unifrom float value or (2) that specified by a list.
    If user does not specify any edge values, this outputs a fully connected random graph using initialize_with_random_beliefs (with seed = 0))"""
    G = nx.complete_graph(N)

    # if user does not input anything
    if edge_values == "default":
        print(
            "Oops, something went wrong in initialising custom belief network. \
                Probably, you didn't input edge values.\
                Resorting to a default random belief network"
        )
        initialize_with_random_beliefs(G, seed=0)
    else:
        if type(edge_values) == float or type(edge_values) == int:
            nx.set_edge_attributes(
                G, {edge: edge_values for edge in G.edges()}, "belief"
            )
        if type(edge_values) == list:
            nx.set_edge_attributes(
                G, {edge: edge_values[i] for i, edge in enumerate(G.edges())}, "belief"
            )
    return G


def triad_energy(G: nx.Graph, triad, weight_key="belief") -> float:
    """Calculate the product of beliefs."""
    triadic_weights = [G[n1][n2][weight_key] for n1, n2 in itertools.combinations(triad, 2)]
    return reduce(mul, triadic_weights, 1)


def internal_energy(G: nx.Graph) -> float:
    """calcualte the internal energy given the belief network using social balance."""
    triads = [c for c in nx.enumerate_all_cliques(G) if len(c) == 3]
    return -1.0 * sum(triad_energy(G, triad) for triad in triads) / len(triads)


def pairwise_social_energy(Gi: nx.Graph, Gj: nx.Graph) -> float:
    """calculate social energy between two belief networks."""
    common_edges = Gi.edges() & Gj.edges()
    #     return -1.0 * sum(
    #         Gi[n1][n2]["belief"] * Gj[n1][n2]["belief"] for n1, n2 in common_edges
    #     )
    return sum(Gi[n1][n2]["belief"] * Gj[n1][n2]["belief"] for n1, n2 in common_edges)


def local_social_energy(
    G_soc: nx.Graph, node: int, ind2bn: dict, kmax: int, belief_size_norm: int
) -> float:
    """calculate social energy for a single node in the social network.

    Parameters
    ----------
    G_soc: nx.Graph
        Social network structure in networkx format.

    belief_size_norm: int
        Social network structure in networkx format.

    ind2bn: dict
        a dictionary that contains individual ID -> belief network: nx.Graph

    Returns
    -------
    float
        social energy.
    """
    neighbors = G_soc.neighbors(node)
    return (
        -1.0
        * sum(
            pairwise_social_energy(ind2bn[node], ind2bn[neighbor])
            for neighbor in neighbors
        )
        / kmax
        / belief_size_norm
    )


def derivative_triad_energy(
    G: nx.Graph, triad, focal_edge, weight_key="belief"
) -> float:
    """Calculate the product of beliefs of a triad not including the focal egde
    Example: (e_int = a*b*c + a*d*e + d*e*f) and focal edge = a
    the derivative of internal energy = b*c + d*e
    """

    # the weights of the triadic edges not including the focal edge
    triadic_weights = [
        G[n1][n2][weight_key]
        for n1, n2 in itertools.combinations(triad, 2)
        if sorted((n1, n2, G[n1][n2][weight_key]))
        != sorted((focal_edge[0], focal_edge[1], focal_edge[2].get("belief")))
    ]
    return reduce(mul, triadic_weights, 1)


def derivative_internal_energy(G: nx.Graph, focal_edge) -> float:

    """calculate the derivative of the internal energy of an individual
    with respect to the focal edge, evaluated at the given weight configuration.
    This is reflects how much the internal energy changes upon changing the focal
    edge weight.

    Parameters
    ----------
    G -> networkx graph of the individual belief network

    focal_edge -> focal "receiver edge" tuple eg: (1, 2, {'belief': -0.9996980509866219})

    Returns
    -------
    float
        derivative of internal evaluated at the given weight configuration (edge weights)
    """
    # The triads not including the ones without the focal edge
    triads = [
        c
        for c in nx.enumerate_all_cliques(G)
        if len(c) == 3 and ((focal_edge[0] in c) and (focal_edge[1] in c))
    ]
    return -1.0 * sum(derivative_triad_energy(G, triad, focal_edge) for triad in triads)


def initialise_star_graph_stabilities(
    hub_kind: str, fixed_neighbors: list, n: int, sn: nx.Graph
) -> dict:

    """set the belief networks of the nodes of a star graph.
        - the belief networks of the fixed_neighbors is the opposite kind as that of the hub

    Parameters
    ----------
    - hub_kind -> 'stable' or 'unstable' indicating the stability of the hub's belief network
    - fixed_neighbors -> node list of neighbors of the hub whose beliefs need to be set to the opposite kind as that of the hub
    - n -> number of concepts (equal across all individuals)
    - sn -> star graph

    Returns
    -------
    bns -> belief networks dictionary -> (edge list which can be understood by networkx as nx.from_edgelist()):
    Example = {node_label:[(edge,{'belief':value})),...]
    bns[0] = [(0, 1, {'belief': 1}), (0, 2, {'belief': 1}), (1, 2, {'belief': -1})], 1: [(0, 1, {'belief': 1}), (0, 2, {'belief': 1}), (1, 2, {'belief': -1})]

    """

    # initialising an empty dictionary to store belief network stabilities(bns)
    bns = {}
    if hub_kind == "stable":
        for i in sn.nodes():
            if i in fixed_neighbors:
                # belief networks of the fixed_neighbors are set to be ustable
                bns[i] = list(complete_belief_network(n, [-1, 1, 1]).edges(data=True))
            else:
                # belief networks of the others are set to be stable
                bns[i] = list(complete_belief_network(n, 1).edges(data=True))

    if hub_kind == "unstable":
        for i in sn.nodes():
            if i in fixed_neighbors:
                # belief networks of the fixed_neighbors are set to be stable
                bns[i] = list(complete_belief_network(n, [1, 1, 1]).edges(data=True))
            else:
                # belief networks of the others are set to be unstable
                bns[i] = list(complete_belief_network(n, [-1, 1, 1]).edges(data=True))
    
    if hub_kind == 'stableplus':
        for i in sn.nodes():
            if i in fixed_neighbors:
                #belief networks of the fixed_neighbors are set to be other kind of stable
                bns[i] = list(complete_belief_network(n, [-1,-1,1]).edges(data=True))
            else:
                #belief networks of the others are set to be stable plus
                bns[i] = list(complete_belief_network(n, [1,1,1]).edges(data=True))

    if hub_kind == 'stableminus':
        for i in sn.nodes():
            if i in fixed_neighbors:
                #belief networks of the fixed_neighbors are set to be other kind of stable
                bns[i] = list(complete_belief_network(n, [1,1,1]).edges(data=True))
            else:
                #belief networks of the others are set to be stable minus
                bns[i] = list(complete_belief_network(n, [-1,-1,1]).edges(data=True))
    return bns

def infer_belief_stability(
        bns
):
    """
    infer the stability of the hub's belief system via  the signs of its beliefs

    Parameters
    ----------

    - bns -> belief networks dictionary -> (edge list which can be understood by networkx as nx.from_edgelist()):
    Example = {node_label:[(edge,{'belief':value})),...]
    bns[0] = [(0, 1, {'belief': 1}), (0, 2, {'belief': 1}), (1, 2, {'belief': -1})], 1: [(0, 1, {'belief': 1}), (0, 2, {'belief': 1}), (1, 2, {'belief': -1})]

    Returns
    -------
    string indicating the hub's stability. 
    One of 'stableplus', 'unstableplus', 'stableminus', 'unstablminus', or 'whatisthis'

    """
    signs = [np.sign(i[2]['belief']) for i in bns[0]]
    count_ones = len([i for i in signs if i==1])
    count_zeros = len([i for i in signs if i==0])
    if count_ones==3 and count_zeros==0:
        return 'stableplus'
    elif count_ones == 2 and count_zeros==0:
        return 'unstableplus'
    elif count_ones == 1 and count_zeros==0:
        return 'stableminus'
    elif count_ones == 0 and count_zeros==0:
        return 'unstablminus'
    else: # should never enter here. 
        # A better error catch can be made but well... 
        return 'whatisthis'

def star_graph_dynamics(
        alpha:float,
        beta:float,
        N:int,
        n:int,
        hub_kind:str
):
    
    """run the weighted belief network dynamics on a STAR GRAPH.

    Parameters
    ----------

    - alpha -> weight of social influence
    - beta -> weight of internal coherence
    - N -> number of individuals in the social network (star graph, in this case)
    - n -> number of concepts in the belief system of all individuals (equal across all individuals)
    - hub_kind -> 'stable', 'unstable', 'stableplus', or 'stableminus' indicating the stability of the hub's belief network

    Returns
    -------
    nested dictionary to store the values necessary for plotting simple vs complex contagion.
    {alpha(float): {beta(float): {fixed_fraction(float):{ensemble_number(int):[time evolution of focal belief of hub (float)]}}}}

    """

    # DEFAULT PARAMS
    normal_scale = 0.2 # variance of the normal distribution from which new beliefs are sampled
    
    # Star graph has N nodes, thus N-1 edges 
    # To give every node a chance to communicate its belief atleast once, 
    # time period of simulation is set to 2*N 
    # since each of the N-1 (approx. N) edges need to be active twice
    # since these are pairwise interactions.   
    T = 2*N 

    data_dict = {}
    data_dict[alpha] = {}
    data_dict[alpha][beta] = {}

    # stability fixed nodes' (a.k.a zealots) belief systems never change during the simulation.
    for stability_fixed_nodes in range(N):


        #0th node is the center (focal). Dont want that to be a zealot
        fixed_receiver = random.sample(range(1,N), stability_fixed_nodes) 
        fraction = stability_fixed_nodes/N
        data_dict[alpha][beta][fraction] = []

        # create the star graph of N nodes. 
        sn = nx.star_graph(N)

        # initialise the stabilities 
        bns = {}
        bns = initialise_star_graph_stabilities(hub_kind,fixed_receiver,n,sn)

        # store starting state of hub
        data_dict[alpha][beta][fraction].append(hub_kind)

        # start the crux of the simulation
        
        for t in range(0,2*N):
            print(t)
            if t%100 == 0:
                print(f"Simulation is in time step {t} out of {2*N}")
            # selecting a random edge of the social network
            # from it extract the sender and receiver of the belief
            random.seed()
            sampled_node_pair = random.sample(sn.edges, 1)
            if random.choice([0,1]) == 0:
                receiver = sampled_node_pair[0][0]
                sender = sampled_node_pair[0][1]
            else:
                sender = sampled_node_pair[0][0]
                receiver = sampled_node_pair[0][1]

            # if receiver is a zealot, nothing happens
            # move onto the next time step
            if receiver in fixed_receiver:
                data_dict[alpha][beta][fraction].append(infer_belief_stability(bns))
                continue

            #getting the belief of a random sender edge
            edge_idx = random.randint(0,2)
            sender_edge = [bns[sender][edge_idx]]
            sender_belief = sender_edge[0][2].get('belief')

            #getting belief of the same edge of the receiver
            receiver_edge = bns[receiver][edge_idx]
            receiver_belief = receiver_edge[2].get('belief')

            #getting the components of the mean of the normal distribution.
            der = -1.0*derivative_internal_energy(nx.from_edgelist(bns[receiver]),receiver_edge)
            mean_ = (alpha*sender_belief) + (beta*der)

            #choosing a random variable from the normal distribution.
            np.random.seed()
            scale = normal_scale

            # update belief
            new_belief = receiver_belief + list(norm.rvs(loc=mean_,scale=scale,size=1))[0]

            #using the linear function to bound the new belief between +1 and -1
            if new_belief > 1:
                new_belief = 1
            elif new_belief < -1:
                new_belief = -1

            bns[receiver][edge_idx][2]['belief'] = new_belief

            data_dict[alpha][beta][fraction].append(infer_belief_stability(bns))

    return data_dict


def community_social_network(N: int, mu: float, M: int, seed=None):

    """Obtain the social network of two communities with half of the nodes in c1 and half in c2.
    - probability of intra community edges = (1-mu)
    - probability of inter community edges = mu

    Parameters
    ----------
    - N -> Number of nodes of the social network
    - mu -> parameter to calculate probabilities of inter and intra community edges
    - M -> Number of edges of the social network

    Returns
    -------
    - The Social Network -> nx.Graph()
    - list of nodes in community 1
    - list of nodes in community 2

    """

    # fraction of intra community links
    intra = math.ceil((1 - mu) * M)

    # fraction of inter community links
    inter = M - intra

    G = nx.Graph()
    G.add_nodes_from(list(range(0, N)))

    if seed is None:
        random.seed()
    else:
        random.seed(seed)

    # community 1 - half the nodes
    comm1 = random.sample(list(range(0, N)), k=math.ceil(N / 2))
    # community 2 - the remaining nodes
    comm2 = list(set(range(0, N)) - set(comm1))

    # intra community edges
    G.add_edges_from(
        random.sample(
            list(itertools.combinations(comm1, 2))
            + list(itertools.combinations(comm2, 2)),
            intra,
        )
    )
    # inter community edges
    G.add_edges_from(random.sample(list(itertools.product(comm1, comm2)), inter))

    return G, comm1, comm2


def initialise_community_stabilities(
    ini_cond: str, fixed_inds: list, n: int, sn: nx.Graph
) -> dict:

    """set the belief networks of the nodes of a communities
        - the belief networks of the fixed_inds is the opposite kind as that of the communities

    Parameters
    ----------
    - ini_cond -> 'stable' or 'unstable' indicating the stability of the two communities
    - fixed_inds -> node list of individuals whose beliefs need to be set to the opposite kind as that of communities
    - n -> number of concepts (equal across all individuals)
    - sn -> graph with communities

    Returns
    -------
    bns -> belief networks dictionary -> (edge list which can be understood by networkx as nx.from_edgelist()):
    Example = {node_label:[(edge,{'belief':value})),...]
    bns[0] = [(0, 1, {'belief': 1}), (0, 2, {'belief': 1}), (1, 2, {'belief': -1})], 1: [(0, 1, {'belief': 1}), (0, 2, {'belief': 1}), (1, 2, {'belief': -1})]

    """

    # initialising an empty dictionary to store belief network stabilities(bns)
    bns = {}
    if ini_cond == "stable":
        for i in sn.nodes():
            if i in fixed_inds:
                # belief networks of the fixed_inds are set to be ustable
                bns[i] = list(complete_belief_network(n, [-1, 1, 1]).edges(data=True))
            else:
                # belief networks of the others are set to be stable
                bns[i] = list(complete_belief_network(n, 1).edges(data=True))

    if ini_cond == "unstable":
        for i in sn.nodes():
            if i in fixed_inds:
                # belief networks of the fixed_inds are set to be stable
                bns[i] = list(complete_belief_network(n, [1, 1, 1]).edges(data=True))
            else:
                # belief networks of the others are set to be unstable
                bns[i] = list(complete_belief_network(n, [-1, 1, 1]).edges(data=True))

    if ini_cond == "neutral":
        for i in sn.nodes():
            if i in fixed_inds:
                # belief networks of the fixed_inds are set to be unstable
                bns[i] = list(complete_belief_network(n, [-1, 1, 1]).edges(data=True))
            else:
                # belief networks of the others are set to be neutral
                bns[i] = list(complete_belief_network(n, [0, 1, 1]).edges(data=True))

    if ini_cond == 'stableplus':
            for i in sn.nodes():
                if i in fixed_inds:
                    #belief networks of the fixed_inds are set to be other kind of stable
                    bns[i] = list(complete_belief_network(n, [-1,-1,1]).edges(data=True))
                else:
                    #belief networks of the others are set to be stable plus
                    bns[i] = list(complete_belief_network(n, [1,1,1]).edges(data=True))

    if ini_cond == 'stableminus':
        for i in sn.nodes():
            if i in fixed_inds:
                #belief networks of the fixed_inds are set to be other kind of stable
                bns[i] = list(complete_belief_network(n, [1,1,1]).edges(data=True))
            else:
                #belief networks of the others are set to be stable minus
                bns[i] = list(complete_belief_network(n, [-1,-1,1]).edges(data=True))
    return bns


# def time_dynamics(sn,bns,alpha,beta,fixed_inds,normal_scale,T):

#     for t in range(0,T):

#         #selecting a random edge of the social network
#         random.seed()
#         sampled_node_pair = random.sample(sn.edges, 1)

#         #selecting dose receiver and dose sender
#         if random.choice([0,1]) == 0:
#             receiver = sampled_node_pair[0][0]
#             sender = sampled_node_pair[0][1]
#         else:
#             receiver = sampled_node_pair[0][1]
#             sender = sampled_node_pair[0][0]

#         if receiver in fixed_inds:
#             continue

#         #making sure that the same focal edge is the one which is passed around in each time step. 
#         edge_idx = 0
#         sender_edge = [bns[sender][edge_idx]]
#         sender_belief = sender_edge[0][2].get('belief')

#         #getting belief of the same edge of the receiver
#         receiver_edge = bns[receiver][edge_idx]
#         receiver_belief = receiver_edge[2].get('belief')


#         if (sender_belief == 0.0) and (receiver_belief == 0.0):
#             continue

#         #getting the components of the mean of the normal distribution.
#         der = -1.0*derivative_internal_energy(nx.from_edgelist(bns[receiver]),receiver_edge)
#         mean_of_normal = (alpha*sender_belief) + (beta*der)

#         #choosing a random veriable from the normal distribution.
#         np.random.seed() #may not be required. 
#         scale = normal_scale
#         delta_belief = list(norm.rvs(loc=mean_of_normal,scale=scale,size=1))[0]

#         #update rule
#         new_belief = receiver_belief + delta_belief

#         #using the linear function to bound the new belief between +1 and -1
#         if new_belief > 1:
#             new_belief = 1
#         elif new_belief < -1:
#             new_belief = -1

#         #updating edge list of the receiver.
#         bns[receiver][edge_idx][2]['belief'] = new_belief