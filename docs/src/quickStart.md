# Quick Start

[Install Julia](https://julialang.org/downloads/)

Clone this repo (make sure to include the --recurse-submodules flag so that the modified nearest neighbours
package gets included).

```console
david@home:~$ git clone --recurse-submodules https://github.com/dpshorten/CoTETE.jl.git
```

make sure that CoTETE.jl/src/ is on your JULIA\_LOAD\_PATH. eg:

```console
david@home:~$ export JULIA_LOAD_PATH=:/home/david/CoTETE.jl/src/
```

Fire up the Julia REPL

```console
david@home:~$ julia
```
You will need to add some prerequisite packages.

!!! tip "Tip for new Julia users"
    The Julia REPL has a nifty feature called *prompt pasting*, which means that it
    will automatically remove the `julia>` prompt when you paste. You can, therefore, just copy and paste the entire block
    below without worrying about these prompts.

```julia
julia> import Pkg
julia> Pkg.add("Distances")
julia> Pkg.add("StaticArrays")
julia> Pkg.add("SpecialFunctions")
julia> Pkg.add("Parameters")
julia> Pkg.add("StatsBase")
```
For the first example, lets estimate the TE between uncoupled homogeneous Poisson processes. This
is covered in section II A of [1].
We first create the source and target processes, each with 10 000 events and with rate 1.

```julia
julia> source = 1e3*rand(Int(1e3));
julia> sort!(source);
julia> target = 1e3*rand(Int(1e3));
julia> sort!(target);
```

We can now estimate the TE between these processes, with history embeddings of length 1.

```julia
julia> import CoTETE
julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);
julia> CoTETE.estimate_TE_from_event_times(parameters, target, source)
```

The answer should be close to 0.
