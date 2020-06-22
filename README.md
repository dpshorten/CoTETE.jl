# CoTETE.jl - Continuous-Time Event-based Transfer Entropy

[**Documentation**](https://dpshorten.github.io/CoTETE.jl/docs/build/index.html)

---

This package allows one to estimate the transfer entropy between event-based time series (such as spike trains or social media post times) in continuous time
(that is, without discretising time into bins).

It contains implementations of the estimator and local permutation scheme presented in
[Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other Event-Based Data](https://doi.org/10.1101/2020.06.16.154377).

If you have any issues using this software, please add an issue here on github, or email me at david.shorten@sydney.edu.au

## Getting Started

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

```julia
julia> import Pkg
julia> Pkg.add("Distances")
julia> Pkg.add("StaticArrays")
julia> Pkg.add("SpecialFunctions")
```
For the first example, lets estimate the TE between uncoupled homogeneous Poisson processes. This
is covered in section II A of [1].
We first create the source and target processes, each with 1 000 events and with rate 1.

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

Let's apply the estimator to a more complex problem. We shall simulate the process described as example B
in [2]. The application of the estimator to this example is covered in section II B of [1].
We create the source process as before

```julia
julia> source = 1e3*rand(Int(1e3));
julia> sort!(source);
```

We create the target as an homogeneous Poisson process with rate 10. This rate has to be larger than the largest rate
achieved by the rate function in our example, so that the thinning algorithm can operate.

```julia
julia> target = 1e3*rand(Int(1e4));
julia> sort!(target);
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
               if rand() < λ/target_rate
               	  push!(new_target, event)
               end
           end
    	   return new_target
       end
julia> target = thin_target(source, target, 10);
```

We can now estimate the TE

```julia
julia> CoTETE.calculate_TE_from_event_times(target, source, 1, 1)
```
The answer should be close to 0.5.

For both of these examples, increasing the number of events in the processes will give estimates closer to the true value.



[1] Shorten, D. P., Spinney, R. E., Lizier, J.T. (2020). [Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other Event-Based Data](https://doi.org/10.1101/2020.06.16.154377). bioRxiv 2020.06.16.154377.

[2] Spinney, R. E., Prokopenko, M., & Lizier, J. T. (2017). [Transfer entropy in continuous time, with applications to jump and neural spiking processes](https://doi.org/10.1103/PhysRevE.95.032319). Physical Review E, 95(3), 032319.
