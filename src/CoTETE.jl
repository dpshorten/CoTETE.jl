module CoTETE

export calculate_TE_from_event_times

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
    AIS_only::Bool = false
    kraskov_noise_level::AbstractFloat = 1e-8
    num_surrogates::Integer = 100
    surrogate_num_samples_ratio::AbstractFloat = 1.0
    k_perm::Integer = 5
end

include("preprocessing.jl")


"""
    calculate_TE_from_event_times(
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

```jldoctest calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> source = sort(1e4*rand(Int(1e4)));

julia> target = sort(1e4*rand(Int(1e4)));

julia> using CoTETE

julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);

julia> TE = CoTETE.calculate_TE_from_event_times(parameters, target, source)
0.0

julia> abs(TE - 0) < 0.05 # For Doctesting purposes
true

```
We can also try increasing the length of the target and source history embeddings
```jldoctest calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> parameters = CoTETE.CoTETEParameters(l_x = 3, l_y = 3);

julia> TE = CoTETE.calculate_TE_from_event_times(parameters, target, source)
0.0

julia> abs(TE - 0) < 0.1 # For Doctesting purposes
true

```

Let's try some other options
```jldoctest calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> using Distances: Cityblock

julia> parameters = CoTETE.CoTETEParameters(l_x = 1,
                                            l_y = 2,
                                            k_global = 3,
                                            auto_find_start_and_num_events = false,
                                            start_event = 100,
                                            num_target_events = 5000,
                                            num_samples_ratio = 2.3,
                                            metric = Cityblock());

julia> TE = CoTETE.calculate_TE_from_event_times(parameters, target, source)
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

```julia calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
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

julia> TE = CoTETE.calculate_TE_from_event_times(parameters, target, source)
0.5076

julia> abs(TE - 0.5076) < 0.05 # For Doctesting purposes
true
```

We can also try extending the length of the target embeddings in order to better resolve this
dependency
```julia calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> parameters = CoTETE.CoTETEParameters(l_x = 3, l_y = 1);

julia> TE = CoTETE.calculate_TE_from_event_times(target, source, 3, 1)
0.5076

julia> abs(TE - 0.5076) < 0.05 # For Doctesting purposes
true
```
"""
function calculate_TE_from_event_times(
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

    TE = CoTETE.calculate_TE_from_preprocessed_data(parameters, preprocessed_data)

    return TE

end

function calculate_AIS_and_surrogates(
    target_events::Array{<:AbstractFloat},
    l_x::Integer;
    auto_find_start_and_num_events::Bool = true,
    num_target_events_cap::Integer = -1,
    start_event::Integer = 1,
    num_target_events::Integer = length(target_events) - start_event,
    num_samples_ratio::AbstractFloat = 1.0,
    k_global::Integer = 5,
    metric::Metric = Euclidean(),
    kraskov_noise_level::AbstractFloat = 1e-8,
    num_surrogates::Integer = 100,
    surrogate_num_samples_ratio::AbstractFloat = 1.0,
)
    preprocessed_data = CoTETE.preprocess_event_times(
        target_events,
        Float64[],
        l_x,
        0,
        auto_find_start_and_num_events = auto_find_start_and_num_events,
        num_target_events_cap = num_target_events_cap,
        num_target_events = num_target_events,
        num_samples_ratio = num_samples_ratio,
        start_event = start_event,
        surrogate_num_samples_ratio = surrogate_num_samples_ratio,
    )

    TE =
        -calculate_TE_from_preprocessed_data(
            preprocessed_data,
            k_global = k_global,
            metric = metric,
            AIS_only = true,
        )

    surrogates = zeros(num_surrogates)
    for i = 1:num_surrogates
        surrogate_preprocessed_data = CoTETE.make_AIS_surrogate(
            target_events,
            preprocessed_data,
            Integer(round(
                surrogate_num_samples_ratio * size(preprocessed_data.representation_joint, 2),
            )),
        )
        surrogates[i] =
            -calculate_TE_from_preprocessed_data(
                surrogate_preprocessed_data,
                k_global = k_global,
                metric = metric,
                AIS_only = true,
            )
    end
    p = 0
    for surrogate in surrogates
        if surrogate >= TE
            p += 1
        end
    end
    #println(TE, " ", mean(surrogates), " ", std(surrogates))
    return TE, surrogates, p
end

"""
    calculate_TE_from_preprocessed_data(parameters::CoTETEParameters, preprocessed_data::PreprocessedData)

calculates the TE using the preprocessed data and the given parameters.

```jldoctest calculate_TE_from_preprocessed_data; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> source = sort(1e4*rand(Int(1e4)));

julia> target = sort(1e4*rand(Int(1e4)));

julia> using CoTETE

julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);

julia> preprocessed_data = CoTETE.preprocess_event_times(parameters, target, source_events = source);

julia> TE = CoTETE.calculate_TE_from_preprocessed_data(parameters, preprocessed_data)
0.0

julia> abs(TE - 0) < 0.05 # For Doctesting purposes
true

```
"""
function calculate_TE_from_preprocessed_data(
    parameters::CoTETEParameters,
    preprocessed_data::PreprocessedData,
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

    TE = 0
    for i = 1:size(preprocessed_data.representation_joint, 2)
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

        maximum_radius_conditionals_from_both_knn_searches =
            max(
                maximum(radii_conditionals_from_knn_search),
                maximum(radii_sampled_conditionals_from_knn_search),
            ) + 1e-6 #TODO make this more well-grounded

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

        if !parameters.AIS_only
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

            maximum_radius_joint_from_both_knn_searches =
                max(
                    maximum(radii_joint_from_knn_search),
                    maximum(radii_sampled_joint_from_knn_search),
                ) + 1e-6

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
                preprocessed_data.sampled_representation_joint[:, indices_sampled_joint_from_radius_search],
            ))

            TE += (
                -(l_x_plus_l_z + l_y)log(radius_joint_inside_first_radius) +
                (l_x_plus_l_z + l_y)log(radius_sampled_joint_inside_first_radius) +
                digamma(size(indices_joint_from_radius_search)[1]) - digamma(size(indices_sampled_joint_from_radius_search)[1])
            )
        end
    end

    if parameters.AIS_only
        TE +=
            size(preprocessed_data.representation_joint, 2) * (
                log(size(representation_conditionals, 2) - 1) -
                log(size(sampled_representation_conditionals, 2))
            )
    end

    return (TE / (preprocessed_data.end_timestamp - preprocessed_data.start_timestamp))

end



end
