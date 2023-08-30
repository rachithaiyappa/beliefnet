"""
This script performs the weighted belief network dynmaics on a watts strogatz graph (WS) and 
tracks the time evolution of fraction of individuals in the population with a "stable_plus" ([+1,+1,+1]) belief system.

The simulations are performed in two network settings, i.e. WS with 
(1) rewiring probility = 1 (random network) and seed individuals picked in a random manner
(2) rewiring probility = 0 (clustered network) and seed individuals picked such that they are neighbors, 
    i.e present in the locally clustered region of the WS network
This setting resembles the Centola 2010, online social contagion experiment. See our main paper for more details 

While the seed set are always "stable_plus" individuals, the rest of the population are either
(1)"unstable_plus"[-1,1,1]---the simple contagion setting (set ini_string = "unstable"), or
(2)"stable_minus"[-1,-1,1]---the complex contagion setting (set ini_string = "dual_stable")

The belief system of the seed individuals are fixed throughout the simulation.

It returns a csv file with the fraction of individuals of different kinds at each time step of the simulation for a clustered or a random network, 
    for differennt values of mu and seed set sizes.
The main paper plots stable_plus_fraction vs time.

Run the following from within the `scripts` directory
julia --project=. wattsstrogatz.jl -a 2.0 -b 1.0 -s 0.2 -e 1

* The description of each of the command line parameters in the above call can be found in /src/beliefnet/function.jl
* --project=. runs the script from within the julia environment contained in Project.toml 

Author: Rachith Aiyappa
"""


# This is main script which takes values of alpha, beta, and standard deviation 
#and performs simulations based on these parameters by varying the modularity (mu) and seed nodes size of the social network.

using MetaGraphs: vertices
using Combinatorics
using Distributions
using Random
using StatsBase
using IterTools
using TickTock
using CSV
using DataFrames
include("../src/beliefnet/function.jl")

#Parameters--------------------------------------------------------------
#number of nodes (N)
N = 100
# average degree of each node in Watts Strogatz network
k = 10 
#time of simulation
T = 2*k*N

#get argumnets from the command line
parsed_args = parse_commandline()
α = parsed_args["alpha"] #weight of social influence
β = parsed_args["beta"] #weight of internal coherence
σ = parsed_args["normalscale"] #standard deviation of the normal
ensembles = parsed_args["ensembles"] #number of ensembles

# probability of rewiring (mu = 0 or 1) and stability fixed nodes range, to be varied during the simulation
mus = LinRange(0,1,2)
stability_fixed_nodes_range = range(1,Int64(floor(N/2)),step=5)
# stability_fixed_nodes_range = range(7,10,step=1)


#--------------------------------------------------------------------------

print("alpha=$(α),beta=$(β)")

#initialise empty data frame to store the data
df = DataFrame(
                mu = Float64[], 
                alpha = Float64[], 
                beta = Float64[], 
                fraction=Float64[], 
                ensemble=Int64[], 
                stable_fraction=Float64[], 
                unstable_fraction = Float64[],
                neutral_fraction = Float64[],
                unstable_minus_fraction = Float64[],
                unstable_plus_fraction = Float64[],
                stable_minus_fraction = Float64[],
                stable_plus_fraction = Float64[],
                time = Int64[]
                )


for e in 1:ensembles
    for mu in mus
        mu = round(mu,digits=2)
        for stability_fixed_nodes in stability_fixed_nodes_range
            print(stability_fixed_nodes)

            #string tag to identify the initial setup
            ini_string = "unstable" # or "dual_stable"

            #fraction of seed nodes
            stability_fixed_nodes_size = round(stability_fixed_nodes/N,digits=3)

            #initialise social network
            sn = watts_strogatz(N,10,mu)

            #get a sample of fixed individuals
            if mu == 1 # random  seeds
                fixed_inds = sample(1:N, Int64(stability_fixed_nodes),replace=false)
            end
            if mu == 0 # clustered seeds
                fixed_inds = sample(1:Int64(stability_fixed_nodes), Int64(stability_fixed_nodes),replace=false)
            end

            #get belief networks of all individuals 
            bns = initialise_graph_stabilities(ini_string,fixed_inds,sn)
            
            # begin simulation
            for t in 1:T
                
                # sample node pair
                sampled_node_pair = sample(collect(edges(sn)),1)
                
                # get send and receiver
                if sample(0:1,1)[1] == 0
                    sender = src(sampled_node_pair[1])
                    receiver = dst(sampled_node_pair[1])
                else
                    sender = dst(sampled_node_pair[1])
                    receiver = src(sampled_node_pair[1])
                end

                # if receiver belongs to the seed set (zealots), do nothing
                if receiver in fixed_inds
                    stable_fraction,unstable_fraction,neutral_fraction,unstable_minus_fraction,unstable_plus_fraction,stable_minus_fraction,stable_plus_fraction = final_fraction(bns)
                    data = Dict(
                        :mu=>mu,
                        :alpha=>α,
                        :beta=>β,
                        :fraction=>stability_fixed_nodes_size,
                        :ensemble=>e,
                        :stable_fraction=>stable_fraction,
                        :unstable_fraction => unstable_fraction,
                        :neutral_fraction=>neutral_fraction,
                        :unstable_minus_fraction => unstable_minus_fraction,
                        :unstable_plus_fraction => unstable_plus_fraction,
                        :stable_minus_fraction => stable_minus_fraction,
                        :stable_plus_fraction => stable_plus_fraction,
                        :time=>t)
                    push!(df, data)
                    continue
                end

                # get a random belief to communicate from sender to receiver
                focal_edge = sample(collect(edges(bns[sender])),1)[1]
                sender_belief = get_belief(bns[sender],focal_edge)
                receiver_belief = get_belief(bns[receiver],focal_edge)

                # update belief of receiver
                der = -1.0*∂E∂b(bns[receiver],focal_edge)
                mean_of_normal = (α*sender_belief) + (β*der)
                Δb =  rand(Normal(mean_of_normal, σ),1)
        
                new_belief = receiver_belief + Δb[1]

                # bound belief between [-1,1]
                if new_belief > 1
                    new_belief = 1
                elseif  new_belief < -1
                    new_belief = -1
                end
                set_prop!(bns[receiver], focal_edge, :weight, new_belief)

                # calculate fraction of belief systems of different kind
                stable_fraction,unstable_fraction,neutral_fraction,unstable_minus_fraction,unstable_plus_fraction,stable_minus_fraction,stable_plus_fraction = final_fraction(bns)
                
                data = Dict(
                    :mu=>mu,
                    :alpha=>α,
                    :beta=>β,
                    :fraction=>stability_fixed_nodes_size,
                    :ensemble=>e,
                    :stable_fraction=>stable_fraction,
                    :unstable_fraction => unstable_fraction,
                    :neutral_fraction=>neutral_fraction,
                    :unstable_minus_fraction => unstable_minus_fraction,
                    :unstable_plus_fraction => unstable_plus_fraction,
                    :stable_minus_fraction => stable_minus_fraction,
                    :stable_plus_fraction => stable_plus_fraction,
                    :time=>t)
                push!(df, data)
            end
            # write data
            CSV.write("StablePlusFraction_vs_Time--alpha$(α)_beta$(β)_N$(N)_avk$(k)_normalscale$(σ).csv", df)
        end
    end
end