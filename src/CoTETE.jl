module CoTETE

export estimate_TE_from_event_times

using Parameters
using Distances: evaluate, colwise, Metric, Chebyshev, Euclidean
using SpecialFunctions: digamma, gamma
using Statistics: mean, std

"""
    struct CoTETEParameters
        l_x::Integer = 0
        l_y::Integer = 0
        l_z::Integer = 0
        auto_find_start_and_num_events::Bool = true
        num_target_events_cap::Integer = -1
        start_event::Integer = 1
        num_target_events::Integer = 0
        num_samples_ratio::AbstractFloat = 1.0
        k_global::Integer = 5
        metric::Metric = Euclidean()
        kraskov_noise_level::AbstractFloat = 1e-8
        num_surrogates::Integer = 100
        surrogate_num_samples_ratio::AbstractFloat = 1.0
        k_perm::Integer = 5
    end

- `l_x::Integer`: The number of intervals in the target process to use in the history embeddings.
  Corresponds to ``l_X`` in [^1].
- `l_y::Integer`: The number of intervals in the source process to use in the history embeddings.
  Corresponds to ``l_Y`` in [^1].
- `l_z::Integer = 0`: The number of intervals in the single conditioning process to use in the
  history embeddings. Corresponds to ``l_{Z_1}`` in [^1].
  !!! info "Single conditioning process"
      Note that although the framework developed in [our paper](https://doi.org/10.1101/2020.06.16.154377)
      considers an arbitrary number of extra
      conditioning processes, at present the framework can only handle a single such process.
      This will change in future releases.
- `auto_find_start_and_num_events::Bool = true`: When set to true, the start event will be set to
  the first event for which there are sufficient preceding events in all processes such that the
  embeddings can be constructed. The number of target events will be set such that all time between
  this first event and the last target event is included.
-  `num_target_events_cap::Integer = -1`
- `start_event::Integer = 1`: only used when `auto_find_start_and_num_events = false`. The index
  of the event in the target process from which to start the analysis.
- `num_target_events::Integer = 0`: only used when
  `auto_find_start_and_num_events = false`. The TE will be calculated on the time series from the
  timestamp of the `start_event`-th event of the target process to the timestamp of the
  `start_event + num_target_events`-th event of the target process.
- `num_samples_ratio::AbstractFloat = 1.0`: Controls the number of samples used to estimate the
  probability density of histories unconditional of the occurrence of events in the target process.
  This number of samples will be `num_samples_ratio * num_target_events`.
  Corresponds to ``N_U/N_X`` in [^1].
- `k_global::Integer = 5`: The number of nearest neighbours to consider in initial searches.
- `metric::Metric = Euclidean()`: The metric to use for nearest neighbour and radius searches.
- `num_surrogates::Integer = 100`:
- `surrogate_num_samples_ratio::AbstractFloat = 1.0`: Controls the number of samples used to
  to construct the alternate set of history embeddings used by our local permutation scheme.
  This number of samples will be `surrogate_num_samples_ratio * num_target_events`.
  Corresponds to ``N_{U, \\textrm{surrogate}}/N_X`` in [^1].
- `k_perm::Integer = 5`: The number of neighbouring source embeddings from which to randomly select
  a replacement embedding in the local permutation scheme.


[^1] Shorten, D. P., Spinney, R. E., Lizier, J.T. (2020).
[Estimating Transfer Entropy in Continuous Time Between Neural Spike Trains or Other Event-Based
Data](https://doi.org/10.1101/2020.06.16.154377). bioRxiv 2020.06.16.154377.

[^2] Spinney, R. E., Prokopenko, M., & Lizier, J. T. (2017).
[Transfer entropy in continuous time, with applications to jump and neural spiking
processes](https://doi.org/10.1103/PhysRevE.95.032319). Physical Review E, 95(3), 032319.
"""
@with_kw struct CoTETEParameters
    l_x::Integer = 0
    l_y::Integer = 0
    l_z::Integer = 0
    auto_find_start_and_num_events::Bool = true
    num_target_events_cap::Integer = -1
    start_event::Integer = 1
    num_target_events::Integer = 0
    num_samples_ratio::AbstractFloat = 1.0
    k_global::Integer = 5
    metric::Metric = Euclidean()
    kraskov_noise_level::AbstractFloat = 1e-8
    num_surrogates::Integer = 100
    surrogate_num_samples_ratio::AbstractFloat = 1.0
    k_perm::Integer = 10
end

include("preprocessing.jl")


"""
    estimate_TE_from_event_times(
      parameters::CoTETEParameters,
      target_events::Array{<:AbstractFloat},
      source_events::Array{<:AbstractFloat};
      conditioning_events::Array{<:AbstractFloat} = Float32[],
    )

Estimate the TE from lists of raw event times.

!!! info "Single conditioning process"
      Note that although the framework developed in [our paper](https://doi.org/10.1101/2020.06.16.154377)
      considers an arbitrary number of extra
      conditioning processes, at present the framework can only handle a single such process.
      This will change in future releases.

# Examples

This example demonstrates estimating the TE between uncoupled homogeneous Poisson processes. This
is covered in section II A of [our paper](https://doi.org/10.1101/2020.06.16.154377).
We first create the source and target processes, each with 10 000 events and with rate 1, before
running the estimator.

```jldoctest estimate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> source = sort(1e4*rand(Int(1e4)));

julia> target = sort(1e4*rand(Int(1e4)));

julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);

julia> TE = CoTETE.estimate_TE_from_event_times(parameters, target, source)
0.0

julia> abs(TE - 0) < 0.05 # For Doctesting purposes
true

```
We can also try increasing the length of the target and source history embeddings
```jldoctest estimate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> parameters = CoTETE.CoTETEParameters(l_x = 3, l_y = 3);

julia> TE = CoTETE.estimate_TE_from_event_times(parameters, target, source)
0.0

julia> abs(TE - 0) < 0.1 # For Doctesting purposes
true

```

Let's try some other options
```jldoctest estimate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> using Distances: Cityblock

julia> parameters = CoTETE.CoTETEParameters(l_x = 1,
                                            l_y = 2,
                                            k_global = 3,
                                            auto_find_start_and_num_events = false,
                                            start_event = 100,
                                            num_target_events = 5000,
                                            num_samples_ratio = 2.3,
                                            metric = Cityblock());

julia> TE = CoTETE.estimate_TE_from_event_times(parameters, target, source)
0.0

julia> abs(TE - 0) < 0.1 # For Doctesting purposes
true
```

The next example applies the estimator to a more complex problem, specifically, the process
described as example B in [Spinney et. al.](https://doi.org/10.1103/PhysRevE.95.032319).
The application of the estimator to this example is covered in
section II B of [our paper](https://doi.org/10.1101/2020.06.16.154377).
We create the source process as before. Howevever, the target process is
originally created as an homogeneous Poisson process with rate 10, before a thinning algorithm
is applied to it, in order to provide the dependence on the source.

```julia estimate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> source = sort(1e4*rand(Int(1e4)));

julia> target = sort(1e4*rand(Int(1e5)));

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

julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);

julia> TE = CoTETE.estimate_TE_from_event_times(parameters, target, source)
0.5076

julia> abs(TE - 0.5076) < 0.05 # For Doctesting purposes
true
```

We can also try extending the length of the target embeddings in order to better resolve this
dependency
```julia estimate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> parameters = CoTETE.CoTETEParameters(l_x = 3, l_y = 1);

julia> TE = CoTETE.estimate_TE_from_event_times(target, source, 3, 1)
0.5076

julia> abs(TE - 0.5076) < 0.05 # For Doctesting purposes
true
```
"""
function estimate_TE_from_event_times(
    parameters::CoTETEParameters,
    target_events::Array{<:AbstractFloat},
    source_events::Array{<:AbstractFloat};
    conditioning_events::Array{<:AbstractFloat} = Float32[],
)

    preprocessed_data = CoTETE.preprocess_event_times(
        parameters,
        target_events,
        source_events = source_events,
        conditioning_events = conditioning_events,
    )

    TE = CoTETE.estimate_TE_from_preprocessed_data(parameters, preprocessed_data)

    return TE

end

"""
    function estimate_TE_and_p_value_from_event_times(
        parameters::CoTETEParameters,
        target_events::Array{<:AbstractFloat},
        source_events::Array{<:AbstractFloat};
        conditioning_events::Array{<:AbstractFloat} = Float32[],
        return_surrogate_TE_values::Bool = false,
    )

calculate the TE and the ``p`` value of it being statistically different from zero.

This example demonstrates estimating the TE and ``p`` value between uncoupled homogeneous Poisson processes.
As the true value of the TE is zero, we expect the ``p`` value to be uniformly disributed between zero and one.

We first create the source and target processes, each with 1 000 events and with rate 1, before
running the estimator and the surrogate generation procedure.

# Examples

```jldoctest estimate_TE_from_event_times; filter = r"\\(.*\\)"
julia> source = sort(1e3*rand(Int(1e3)));

julia> target = sort(1e3*rand(Int(1e3)));

julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);

julia> TE, p = CoTETE.estimate_TE_and_p_value_from_event_times(parameters, target, source)
(0.0, 0.5)

julia> p > 0.05 # For Doctesting purposes. Should fail every now and then.
true

```

This second example shows using this method on coupled processes for which the true value of the
TE is nonzero. As there is a strong coupling between the source and target, we expect the
``p`` value to be close to 0.
The application of the estimator to this example is covered in
section II B of [our paper](https://doi.org/10.1101/2020.06.16.154377). See the above examples for
`estimate_TE_from_event_times` for more details as well as the implementation of the thinning algorithm.

We create the source process as before. Howevever, the target process is
originally created as an homogeneous Poisson process with rate 10, before the thinning algorithm
is applied to it, in order to provide the dependence on the source.

```julia estimate_TE_from_event_times; filter = r"\\(.*\\)"
julia> source = sort(1e3*rand(Int(1e3)));

julia> target = sort(1e3*rand(Int(1e4)));

julia> target = thin_target(source, target, 10);

julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);

julia> TE, p = CoTETE.estimate_TE_and_p_value_from_event_times(parameters, target, source)
(0.5, 0.01)

julia> p < 0.05 # For Doctesting purposes. Should fail very rarely.
true
```

"""
function estimate_TE_and_p_value_from_event_times(
    parameters::CoTETEParameters,
    target_events::Array{<:AbstractFloat},
    source_events::Array{<:AbstractFloat};
    conditioning_events::Array{<:AbstractFloat} = Float32[],
    return_surrogate_TE_values::Bool = false,
)

    preprocessed_data = CoTETE.preprocess_event_times(
        parameters,
        target_events,
        source_events = source_events,
        conditioning_events = conditioning_events,
    )

    TE = CoTETE.estimate_TE_from_preprocessed_data(parameters, preprocessed_data)

    surrogate_TE_values = zeros(parameters.num_surrogates)
    for i = 1:parameters.num_surrogates
        surrogate_preprocessed_data = deepcopy(preprocessed_data)
        CoTETE.make_surrogate!(
            parameters,
            surrogate_preprocessed_data,
            target_events,
            source_events,
            conditioning_events = conditioning_events,
        )
        surrogate_TE_values[i] =
            CoTETE.estimate_TE_from_preprocessed_data(parameters, surrogate_preprocessed_data)
    end

    p = 0
    for surrogate_val in surrogate_TE_values
        if surrogate_val >= TE
            p += 1
        end
    end
    p /= parameters.num_surrogates

    if return_surrogate_TE_values
        return TE, p, surrogate_TE_values
    else
        return TE, p
    end

end

"""
    estimate_AIS_from_event_times(
      parameters::CoTETEParameters,
      target_events::Array{<:AbstractFloat},
      source_events::Array{<:AbstractFloat};
      conditioning_events::Array{<:AbstractFloat} = Float32[],
    )

Estimate the Active Information Storage (AIS) from lists of raw event times.

See [this thesis](https://doi.org/10.1007/978-3-642-32952-4) for a description of AIS.

# Examples

This example estimates the AIS on an homogeneous Poisson process. The true value of the
AIS on such a process is zero.

```jldoctest; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> target = sort(1e4*rand(Int(1e4)));

julia> parameters = CoTETE.CoTETEParameters(l_x = 1);

julia> AIS = CoTETE.estimate_AIS_from_event_times(parameters, target)
0.0

julia> abs(AIS - 0) < 0.05 # For Doctesting purposes
true

```
"""
function estimate_AIS_from_event_times(
    parameters::CoTETEParameters,
    target_events::Array{<:AbstractFloat},
)

    preprocessed_data = CoTETE.preprocess_event_times(parameters, target_events)

    AIS = -CoTETE.estimate_TE_from_preprocessed_data(parameters, preprocessed_data, AIS_only = true)

    return AIS

end

"""
    estimate_AIS_and_p_value_from_event_times(
      parameters::CoTETEParameters,
      target_events::Array{<:AbstractFloat},
      source_events::Array{<:AbstractFloat};
      conditioning_events::Array{<:AbstractFloat} = Float32[],
    )

Estimate the Active Information Storage (AIS) along with the ``p`` value of the AIS being different
from 0 from lists of raw event times.

See [this thesis](https://doi.org/10.1007/978-3-642-32952-4) for a description of AIS.

# Examples

This example estimates the AIS and ``p`` value on an homogeneous Poisson process. The true value of the
AIS on such a process is zero. We expect the ``p`` value to be uniformly distributed between zero and 1.

```jldoctest;  filter = r"\\(.*\\)"
julia> target = sort(1e3*rand(Int(1e3)));

julia> parameters = CoTETE.CoTETEParameters(l_x = 1);

julia> AIS, p = CoTETE.estimate_AIS_and_p_value_from_event_times(parameters, target)
(0.0, 0.5)

julia> p > 0.05 # For Doctesting purposes. Should fail from time to time
true

```
This next example estimates the AIS for a process where we know that the AIS must be nonzero. This process has an event
occurring every one time unit, with a bit of noise added to the event times.
```jldoctest;  filter = r"\\(.*\\)"
julia> target = sort(cumsum(ones(Int(1e3))) .+ 1e-2*randn(Int(1e3)));

julia> parameters = CoTETE.CoTETEParameters(l_x = 1);

julia> AIS, p = CoTETE.estimate_AIS_and_p_value_from_event_times(parameters, target)
(1.0, 0.01)

julia> p < 0.05 # For Doctesting purposes. Should fail from time to time
true

```
"""
function estimate_AIS_and_p_value_from_event_times(
    parameters::CoTETEParameters,
    target_events::Array{<:AbstractFloat},
    return_surrogate_AIS_values::Bool = false,
)

    preprocessed_data = CoTETE.preprocess_event_times(parameters, target_events)

    AIS = -CoTETE.estimate_TE_from_preprocessed_data(parameters, preprocessed_data, AIS_only = true)

    surrogate_AIS_values = zeros(parameters.num_surrogates)
    for i = 1:parameters.num_surrogates
        surrogate_preprocessed_data = deepcopy(preprocessed_data)
        CoTETE.make_AIS_surrogate!(parameters, surrogate_preprocessed_data, target_events)
        surrogate_AIS_values[i] =
            -CoTETE.estimate_TE_from_preprocessed_data(parameters, surrogate_preprocessed_data, AIS_only = true)
    end

    p = 0
    for surrogate_val in surrogate_AIS_values
        if surrogate_val >= AIS
            p += 1
        end
    end
    p /= parameters.num_surrogates

    if return_surrogate_AIS_values
        return AIS, p, surrogate_AIS_values
    else
        return AIS, p
    end

end

"""
    estimate_TE_from_preprocessed_data(parameters::CoTETEParameters, preprocessed_data::PreprocessedData)

calculates the TE using the preprocessed data and the given parameters.

```jldoctest estimate_TE_from_preprocessed_data; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> source = sort(1e4*rand(Int(1e4)));

julia> target = sort(1e4*rand(Int(1e4)));

julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);

julia> preprocessed_data = CoTETE.preprocess_event_times(parameters, target, source_events = source);

julia> TE = CoTETE.estimate_TE_from_preprocessed_data(parameters, preprocessed_data)
0.0

julia> abs(TE - 0) < 0.05 # For Doctesting purposes
true

```
"""
function estimate_TE_from_preprocessed_data(
    parameters::CoTETEParameters,
    preprocessed_data::PreprocessedData;
    AIS_only::Bool = false,
)

    # Lets declare these to make the rest of this function less verbose
    l_x_plus_l_z = parameters.l_x + parameters.l_z
    l_y = parameters.l_y

    # Add a bit of noise as recommended Kraskov
    preprocessed_data.representation_joint[:, :] +=
        parameters.kraskov_noise_level .* randn(size(preprocessed_data.representation_joint))
    preprocessed_data.sampled_representation_joint[:, :] +=
        parameters.kraskov_noise_level .*
        randn(size(preprocessed_data.sampled_representation_joint))

    # Pull out the conditionals from the joint
    representation_conditionals = preprocessed_data.representation_joint[1:(l_x_plus_l_z), :]
    sampled_representation_conditionals =
        preprocessed_data.sampled_representation_joint[1:(l_x_plus_l_z), :]


    tree_joint = NearestNeighbors.KDTree(
        preprocessed_data.representation_joint,
        parameters.metric,
        reorder = false,
    )

    tree_sampled_joint = NearestNeighbors.KDTree(
        preprocessed_data.sampled_representation_joint,
        parameters.metric,
        reorder = false,
    )

    tree_conditionals =
        NearestNeighbors.KDTree(representation_conditionals, parameters.metric, reorder = false)

    tree_sampled_conditionals = NearestNeighbors.KDTree(
        sampled_representation_conditionals,
        parameters.metric,
        reorder = false,
    )

    TE = 0.0
    # Iterate over each event in the target process
    for i = 1:size(preprocessed_data.representation_joint, 2)
        #=
          We first estimate the contribution from the AIS. This corresponeds to the second
          KL divergence term in equation 9 of doi.org/10.1101/2020.06.16.154377.
        =#
        indices_conditionals_from_knn_search, radii_conditionals_from_knn_search =
            NearestNeighbors.knn(
                tree_conditionals,
                representation_conditionals[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.exclusion_windows,
                parameters.k_global,
            )

        indices_sampled_conditionals_from_knn_search, radii_sampled_conditionals_from_knn_search =
            NearestNeighbors.knn(
                tree_sampled_conditionals,
                representation_conditionals[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.sampled_exclusion_windows,
                parameters.k_global,
            )

        maximum_radius_conditionals_from_both_knn_searches = max(
            maximum(radii_conditionals_from_knn_search),
            maximum(radii_sampled_conditionals_from_knn_search),
        )

        indices_conditionals_from_radius_search = NearestNeighbors.inrange(
            tree_conditionals,
            representation_conditionals[:, i],
            preprocessed_data.exclusion_windows[:, :, i],
            preprocessed_data.exclusion_windows,
            maximum_radius_conditionals_from_both_knn_searches,
        )

        radius_conditionals_inside_first_radius = maximum(colwise(
            parameters.metric,
            representation_conditionals[:, i],
            representation_conditionals[:, indices_conditionals_from_radius_search],
        ))

        indices_sampled_conditionals_from_radius_search = NearestNeighbors.inrange(
            tree_sampled_conditionals,
            representation_conditionals[:, i],
            preprocessed_data.exclusion_windows[:, :, i],
            preprocessed_data.sampled_exclusion_windows,
            maximum_radius_conditionals_from_both_knn_searches,
        )

        radius_sampled_conditionals_inside_first_radius = maximum(colwise(
            parameters.metric,
            representation_conditionals[:, i],
            sampled_representation_conditionals[:, indices_sampled_conditionals_from_radius_search],
        ))

        TE += (
            l_x_plus_l_z * log(radius_conditionals_inside_first_radius) -
            l_x_plus_l_z * log(radius_sampled_conditionals_inside_first_radius) -
            digamma(size(indices_conditionals_from_radius_search)[1]) +
            digamma(size(indices_sampled_conditionals_from_radius_search)[1])
        )

        if !AIS_only
            #=
              We now estimate the contribution from the first
              KL divergence term in equation 9 of doi.org/10.1101/2020.06.16.154377.
            =#
            indices_joint_from_knn_search, radii_joint_from_knn_search = NearestNeighbors.knn(
                tree_joint,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.exclusion_windows,
                parameters.k_global,
            )

            indices_sampled_joint_from_knn_search, radii_sampled_joint_from_knn_search =
                NearestNeighbors.knn(
                    tree_sampled_joint,
                    preprocessed_data.representation_joint[:, i],
                    preprocessed_data.exclusion_windows[:, :, i],
                    preprocessed_data.sampled_exclusion_windows,
                    parameters.k_global,
                )

            maximum_radius_joint_from_both_knn_searches = max(
                maximum(radii_joint_from_knn_search),
                maximum(radii_sampled_joint_from_knn_search),
            )

            indices_joint_from_radius_search = NearestNeighbors.inrange(
                tree_joint,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.exclusion_windows,
                maximum_radius_joint_from_both_knn_searches,
            )

            radius_joint_inside_first_radius = maximum(colwise(
                parameters.metric,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.representation_joint[:, indices_joint_from_radius_search],
            ))

            indices_sampled_joint_from_radius_search = NearestNeighbors.inrange(
                tree_sampled_joint,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.sampled_exclusion_windows,
                maximum_radius_joint_from_both_knn_searches,
            )


            radius_sampled_joint_inside_first_radius = maximum(colwise(
                parameters.metric,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.sampled_representation_joint[
                    :,
                    indices_sampled_joint_from_radius_search,
                ],
            ))

            TE += (
                -(l_x_plus_l_z + l_y)log(radius_joint_inside_first_radius) +
                (l_x_plus_l_z + l_y)log(radius_sampled_joint_inside_first_radius) +
                digamma(size(indices_joint_from_radius_search)[1]) -
                digamma(size(indices_sampled_joint_from_radius_search)[1])
            )
        end
    end

    #AIS correction as the log(N_X) and log(N_U) terms no longer cancel.
    if AIS_only
        TE +=
            size(preprocessed_data.representation_joint, 2) * (
                log(size(representation_conditionals, 2) - 1) -
                log(size(sampled_representation_conditionals, 2))
            )
    end

    return (TE / (preprocessed_data.end_timestamp - preprocessed_data.start_timestamp))

end



end
