# CoTETE.jl

Continuous-Time Event-based Transfer Entropy

[Documentation](https://dpshorten.github.io/CoTETE.jl/docs/build/index.html)

## Getting Started

[Install Julia](https://julialang.org/downloads/)

Clone this repo (make sure to include the --recurse-submodules flag so that the modified nearest neighbors
package gets included).

```console
david@home:~$ git clone --recurse-submodules https://github.com/dpshorten/CoTETE.jl.git
```

make sure that CoTETE.jl/src/ is on your JULIA_LOAD_PATH. eg:

```console
david@home:~$ export JULIA_LOAD_PATH=:/home/david/CoTETE.jl/src/
```

Fire up the Julia REPL

```console
david@home:~$ julia
```

You will need to add three prerequisite packages.

```julia
julia> import Pkg; Pkg.add("Distances"); Pkg.add("StaticArrays"); Pkg.add("SpecialFunctions")
```

Let's now create a source and a target as homogeneous Poisson processes, each with 1000 events.

```julia
julia> source = 1e3*rand(Int(1e3)); sort!(source);
julia> target = 1e3*rand(Int(1e3)); sort!(target);
```

We can now estimate the TE between these processes, with history embeddings of length 1.

```julia
julia> import CoTETE
julia> CoTETE.do_preprocessing_and_calculate_TE(target, source, 1, 1, start_event = 10)
```

The answer should be close to 0.