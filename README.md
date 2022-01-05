# CoTETE.jl
*Continuous-Time Event-based Transfer Entropy*

Contains an implementation of the estimator proposed in [this paper](https://doi.org/10.1371/journal.pcbi.1008054)

It is easy to call this package from **Python**. See this [tutorial](https://dpshorten.github.io/CoTETE.jl/docs/build/quickStartPython/) for a quick guide on how to do this.

### [Documentation](https://dpshorten.github.io/CoTETE.jl/docs/build/index.html)

## Introduction

Transfer entropy (TE) is a measure of information flow between time series. It can be used to
infer functional and effective networks.

This package allows one to estimate the TE between event-based time series
(such as spike trains or social media post times) in continuous time (that is, without discretising
time into bins). The advantages of this approach over the discrete-time approach include:
* The continuous-time approach is **provably consistent** - it is guaranteed to converge to the true
  value of the TE in the limit of infinite data. The discrete-time estimator is not consistent. It is easy to create examples
  where it does not converge to the true value of the TE.
* The discrete-time approach is thwarted by having an effective limit on the total number of bins
  that can be used for history embeddings. This means that the user of this approach must choose between
  capturing relationships occurring over long time intervals, or those that occurr with fine time precision
  (by choosing either a large or small bin size).
  They can never capture both simultaneously. By contrast, the continuous-time approach can capture
  relationships occurring over relatively long time intervals with **no loss of precision**.
* On synthetic examples studied, the continuous-time approach **converges orders of magnitude faster**
  than the discrete-time approach and exhibits substantially lower bias.
* In the inference of structural and functional connectivity, the discrete-time approach was typically
  coupled with a surrogate generation method which utilised an incorrect null hypothesis. The
  use of this method can be demonstrated to lead to high false-positive rates.
  CoTETE.jl contains an implementation of a method for generating surrogates which conform to the
  correct null hypothesis of conditional independence.
See [our paper](https://doi.org/10.1371/journal.pcbi.1008054) for more details on all of these points.

Transfer entropy has already been widely applied to the spiking activity of neurons.
Notable work on the application of TE to spike trains include:
* [The reconstruction](https://doi.org/10.1371/journal.pcbi.1002653) of the
  structural connectivity of neurons from simulated calcium imaging data.
  [See here](https://doi.org/10.1371/journal.pone.0098842) for an extension to this work.
* The inference of structural connectivity from models of spiking neural networks
  ([1](https://doi.org/10.1007/s10827-013-0443-y), [2](https://doi.org/10.1371/journal.pone.0027431)).
* [Investigation](https://doi.org/10.1371/journal.pcbi.1007226) of the energy efficiency of
  synaptic information transfer.
* The inference of functional and effective networks (
  [1](https://doi.org/10.1523/jneurosci.2177-15.2016),
  [2](https://doi.org/10.1371/journal.pone.0115764),
  [3](https://doi.org/10.1371/journal.pcbi.1004858),
  [4](https://doi.org/10.1103/PhysRevE.90.022721)
  )

## Getting Started

[Install Julia](https://julialang.org/downloads/)

Clone this repo (make sure to include the --recurse-submodules flag so that the modified nearest neighbours
package gets included as well as the --branch flag to avoid recent potentially unstable changes).

```console
david@home:~$ git clone --recurse-submodules --branch v0.2.1 https://github.com/dpshorten/CoTETE.jl.git
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

>**Note for new Julia users:** The Julia REPL has a nifty feature called *prompt pasting*, which means that it
> will automatically remove the `julia>` prompt when you paste. You can, therefore, just copy and paste the entire block
> below without worrying about these prompts.

```julia
julia> import Pkg
julia> Pkg.add("Distances")
julia> Pkg.add("StaticArrays")
julia> Pkg.add("SpecialFunctions")
julia> Pkg.add("Parameters")
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
julia> CoTETE.estimate_TE_from_event_times(target, source, 1, 1)
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
julia> CoTETE.estimate_TE_from_event_times(target, source, 1, 1)
```
The answer should be close to 0.5.

For both of these examples, increasing the number of events in the processes will give estimates closer to the true value.

## Note on negative values

Kraskov-style estimators of information-theoretic quantities (such as this one) can produce negative values. In TE estimation this is most commonly encountered when the target present state has a strong dependence on the target history but is only weakly dependent (or conditionally independent) of the source history. This leads to a violation of the assumption of local uniformity and a negative bias. This issue is discussed in detail in the 9th paragraph of the Discussion section of [the paper](https://doi.org/10.1371/journal.pcbi.1008054) proposing this estimator. A good discussion of it can also be found in the [JIDT documentation](https://github.com/jlizier/jidt/wiki/FAQs#what-does-it-mean-if-i-get-negative-results-from-a-kraskov-stoegbauer-grassberger-estimator). As mentioned in these resources, the issue can be easily resolved by debiasing the estimator by subtracting the mean of the surrogate TE estimates from the estimated value. This debiasing procedure should be incorporated as an option in this library in the near future.

## Assistance

If you have any issues using this software, please add an issue here on github, or email me at david.shorten@sydney.edu.au

## References

[1] Shorten, D. P., Spinney, R. E., Lizier, J.T. (2021). [Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other Event-Based Data](https://doi.org/10.1371/journal.pcbi.1008054). PLoS Computaional Biology.

[2] Spinney, R. E., Prokopenko, M., & Lizier, J. T. (2017). [Transfer entropy in continuous time, with applications to jump and neural spiking processes](https://doi.org/10.1103/PhysRevE.95.032319). Physical Review E, 95(3), 032319.
