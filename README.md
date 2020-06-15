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
julia> import Pkg
julia> Pkg.add("Distances")
julia> Pkg.add("StaticArrays")
julia> Pkg.add("SpecialFunctions")
```

Let's now create a source and a target as homogeneous Poisson processes, each with 1000 events and with rate 1.

```julia
julia> source = 1e3*rand(Int(1e3));
julia> sort!(source);
julia> target = 1e3*rand(Int(1e3));
julia> sort!(target);
```

We can now estimate the TE between these processes, with history embeddings of length 1.

```julia
julia> import CoTETE
julia> CoTETE.do_preprocessing_and_calculate_TE(target, source, 1, 1, start_event = 10)
```

The answer should be close to 0.

Let's apply the estimator to a more complex problem. We shall simulate the process described as example B
in [1]. We create the source process as before

```julia
julia> source = 1e3*rand(Int(1e3));
julia> sort!(source);
```

We create the target as an homogeneous Poisson process with rate 10. This rate has to be larger than the largest rate
achieved by the rate function in our example, so that the thinning algorithm can operate.

```julia
julia> source = 1e3*rand(Int(1e4));
julia> sort!(source);
```

We then run the thinning algorithm on the target.

```julia
julia> function thin_target(source, target, target_rate)
           # Remove target events occurring before first source
    	   start_index = 1
    	   while target[start_index] < source[1]
           	 start_index += 1
    	   end
    	   target = target[start_index:end]
    
	   new_target = Float64[]
    	   index_of_last_source = 1
    	   for event in target
               while index_of_last_source < length(source) && source[index_of_last_source + 1] < event
               	     index_of_last_source += 1
               end
               distance_to_last_source = event - source[index_of_last_source]
               λ = 0.5 + 5exp(-50(distance_to_last_source - 0.5)^2) - 5exp(-50(-0.5)^2)
               if rand() < λ/10
               	  push!(new_target, event)
               end
           end
    	   return new_target
       end
julia> target = thin_target(source, target, target_rate)
```

We can now estimate the TE

```julia
julia> CoTETE.do_preprocessing_and_calculate_TE(target, source, 1, 1, start_event = 10)
```
The answer should be close to 0.5.


[1] Spinney, R. E., Prokopenko, M., & Lizier, J. T. (2017). Transfer entropy in continuous time, with applications to jump and neural spiking processes. Physical Review E, 95(3), 032319.