# Quick Start

[Install Julia](https://julialang.org/downloads/)

Clone this repo (make sure to include the --recurse-submodules flag so that the modified nearest neighbours
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

!!! tip "Tip for new Julia users"
    The Julia REPL has a nifty feature called *prompt pasting*, which means that it
    will automatically remove the `julia>` prompt when you paste. You can, therefore, just copy and paste the entire block
    below without worrying about these prompts.

```julia
julia> import Pkg
julia> Pkg.add("Distances")
julia> Pkg.add("StaticArrays")
julia> Pkg.add("SpecialFunctions")
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
julia> CoTETE.calculate_TE_from_event_times(target, source, 1, 1)
```

The answer should be close to 0.
