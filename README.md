# beliefnet

A persistent repository where the scripts for the belief network project will be stored and shared

This repository hosts the main part of the code which led to the results in our paper [Weighted Belief Networks Unify Simple and Complex Contagion Dynamics](https://arxiv.org/pdf/2301.02368.pdf)
Figure 2 of the main paper was obtained using a python script `scripts/stargraph.py`
Figure 3, was obtained using a julia scirpt `scripts/wattsstrogatz.jl` since python was extremely slow for our simulations

## Setting up
### Steps to get the **python scripts** of this repo working

1. `git clone git@github.com:rachithaiyappa/beliefnet.git`
2. `cd beliefnet`
3. `conda env create --name beliefnet --file=environment.yml`
4. `cd src`
5. `pip install -e .`

These steps creates the environment and installs the belief network package which has some useful functions to do...stuff. Check out `src/beliefet/model` to know more. 

*Makefile coming up soon*

Incase the environment.yml fails to build for you, the required packages in this repo are:
1. python = 3.7.9
2. numpy = 1.19.5
3. networkx = 2.5

However, I have tested the environment.yml on linux and OSx machines. It builds. 
I'd avoid trying to setup your own environment from scratch. 
Incase the enviroment does not build because, for some reason, conda cannot fetch some of the packages listed in environment.yml, I'd suggest deleting those pacakges from the environment.yml and trying to rerun step 3 above. 

#### Examples

An example script from which one run of Fig 2c. (orange) can be reproduced is shown in `scripts/stargraph.py`

### Steps to get the **julia scripts** of this repo working

1. Download [julia 1.4](https://julialang.org/downloads/oldreleases/) and make sure you add it to your PATH environmental variable such that typing in `julia` from any directory in your terminal should call it
2. cd scripts
3. The command line call for each of the Julia scripts can be found in the script's description. 
