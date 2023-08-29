using LightGraphs
using MetaGraphs
using Combinatorics
using Distributions
using Random
using StatsBase
using IterTools
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--alpha", "-a"
            help = "Weight of social influence"
            arg_type = Float64
            required = true
        "--beta", "-b"
            help = "Weight of internal coherence"
            arg_type = Float64
            required = true
        "--normalscale", "-s"
            help = "Standard Deviation of Normal Distribution"
            arg_type = Float64
            required = true
        "--ensembles", "-e"
            help = "Number of ensembles"
            arg_type = Int
            required = true
    end

    return parse_args(s)
end

function get_belief(g, edge)
	return get_prop(g, edge, :weight)
end

function ∂E∂b(g, focal_edge)
    """derivative of energy with respect to a focal belief"""
    
    eweights = [get_prop(g,e,:weight) for e in edges(g) if focal_edge != e]
	return -1.0*reduce(*, eweights)
end

function create_triad_belief_net(edge_weights)
    """create a triadic belief network"""

    g = MetaGraph(3)
    add_edge!(g, 1, 2)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 3)
    set_prop!(g, Edge(1,2), :weight, edge_weights[1])
    set_prop!(g, Edge(1,3), :weight, edge_weights[2])
    set_prop!(g, Edge(2,3), :weight, edge_weights[3])
    return g
end

function initialise_graph_stabilities(ini_cond :: String, fixed_inds:: Array,sn:: SimpleGraph)
    """the belief systems of each individual in the social network"""

    #initialising an empty dictionary to store belief network stabilities(bns)
    bns = Dict()

    if ini_cond == "dual_stable" #community is stable. Seeds are stable of other kind.
        for i in vertices(sn)
            if i in fixed_inds
                #belief networks of the fixed_inds are set to be stable
                bns[i] = create_triad_belief_net([1.0,1.0,1.0])
            else
                #belief networks of the others are set to be stable
                bns[i] = create_triad_belief_net([1.0,-1.0,-1.0])
            end
        end
    end
            
    if ini_cond == "unstable" #community unstable
        for i in vertices(sn)
            if i in fixed_inds
                #belief networks of the fixed_inds are set to be stable
                bns[i] = create_triad_belief_net([1.0,1.0,1.0])
            else
                #belief networks of the others are set to be unstable
                bns[i] = create_triad_belief_net([-1.0,1.0,1.0])
            end
        end
    end
    return bns
end

function final_fraction(bns)

    """fraction of belief systems of various kinds"""

    stable_count = 0 
    stable_fraction = 0 # [1,1,1] + [-1,-1,1]
    unstable_fraction = 0 # [-1,1,1] + [-1,-1,-1]
    unstable_count = 0
    neutral_count = 0
    neutral_fraction = 0
    unstable_minus = 0  # [-1,-1,-1]
    unstable_plus = 0 # [-1,1,1]
    stable_minus = 0 # [-1,-1,1]
    stable_plus = 0 # [1,1,1]

    unstable_minus_fraction,unstable_plus_fraction,stable_minus_fraction,stable_plus_fraction  = 0,0,0,0

    for (node, graph) in bns
        eweights = [get_prop(bns[node],e,:weight) for e in edges(bns[node])]
        prod = reduce(*,eweights)
        if prod < 0
            unstable_count = unstable_count + 1
            signs = [e for e in [sign(edge) for edge in eweights] if e == -1]
            if length(signs) == 3
                unstable_minus = unstable_minus + 1
            elseif length(signs) == 1
                unstable_plus = unstable_plus + 1
            else
                println("You are not capturing this condition")
            end 
        end
        if prod > 0
            stable_count = stable_count + 1
            signs = [e for e in [sign(edge) for edge in eweights] if e == -1]
            if length(signs) == 2
                stable_minus = stable_minus + 1
            elseif length(signs) == 0
                stable_plus = stable_plus + 1
            else
                println("You are not capturing this condition")
            end
        end
        if prod == 0
            neutral_count = neutral_count + 1
        end
    end 
    stable_fraction = stable_count/length(bns)
    unstable_fraction = unstable_count/length(bns)
    neutral_fraction = neutral_count/length(bns)

    unstable_minus_fraction =  unstable_minus/length(bns)
    unstable_plus_fraction = unstable_plus/length(bns)
    stable_minus_fraction = stable_minus/length(bns)
    stable_plus_fraction = stable_plus/length(bns)
    return stable_fraction,unstable_fraction,neutral_fraction,unstable_minus_fraction,unstable_plus_fraction,stable_minus_fraction,stable_plus_fraction
end