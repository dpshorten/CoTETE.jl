module CoTETE

export calculate_TE_from_event_times

using Distances: evaluate, colwise, Metric, Chebyshev, Euclidean
using SpecialFunctions: digamma, gamma

include("preprocessing.jl")

struct NonZeroTEInferenceResult

"""
    function calculate_TE_from_event_times(
        target_events::Array{<:AbstractFloat},
        source_events::Array{<:AbstractFloat},
        l_x::Integer,
        l_y::Integer;
        auto_find_start_and_num_events::Bool = true,
        start_event::Integer = 1,
        num_target_events::Integer = length(target_events) - start_event,
        num_samples_ratio::AbstractFloat = 1.0,
        k_global::Integer = 5,
        conditioning_events::Array{<:AbstractFloat} = [0.0],
        l_z::Integer = 0,
        metric::Metric = Euclidean(),
        is_surrogate::Bool = false,
        surrogate_num_samples_ratio::AbstractFloat = 1.0,
        k_perm::Integer = 5,
        )

Estimates the TE from lists of raw event times.

# Examples

This example demonstrates estimating the TE between uncoupled homogeneous Poisson processes. This
is covered in section II A of [our paper](https://doi.org/10.1101/2020.06.16.154377).
We first create the source and target processes, each with 10 000 events and with rate 1, before
running the estimator.
```jldoctest calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> source = sort(1e4*rand(Int(1e4)));

julia> target = sort(1e4*rand(Int(1e4)));

julia> using CoTETE

julia> TE = CoTETE.calculate_TE_from_event_times(target, source, 1, 1)
0.0

julia> abs(TE - 0) < 0.05 # For Doctesting purposes
true

```

We can also try increasing the length of the target and source history embeddings
```jldoctest calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> TE = CoTETE.calculate_TE_from_event_times(target, source, 3, 3)
0.0

julia> abs(TE - 0) < 0.1 # For Doctesting purposes
true

```

Let's try some other options
```jldoctest calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> using Distances: Cityblock

julia> TE = CoTETE.calculate_TE_from_event_times(target, source, 1, 2, k_global = 3,
                                                 auto_find_start_and_num_events = false,
                                                 start_event = 100,
                                                 num_target_events = 5000,
                                                 num_samples_ratio = 2.3,
                                                 metric = Cityblock())
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
julia> TE = CoTETE.calculate_TE_from_event_times(target, source, 1, 1)
0.5076

julia> abs(TE - 0.5076) < 0.05 # For Doctesting purposes
true
```

We can also try extending the length of the target embeddings in order to better resolve this
dependency
```julia calculate_TE_from_event_times; filter = r"-?([0-9]+.[0-9]+)|([0-9]+e-?[0-9]+)"
julia> TE = CoTETE.calculate_TE_from_event_times(target, source, 3, 1)
0.5076

julia> abs(TE - 0.5076) < 0.05 # For Doctesting purposes
true
```

# Arguments
- `target_events::Array{<:AbstractFloat}`: A list of the raw event times in the target
  process. Corresponds to ``X`` in [^1].
- `source_events::Array{<:AbstractFloat}`: A list of the raw event times in the source.
  Corresponds to ``Y`` in [^1].
- `l_x::Integer`: The number of intervals in the target process to use in the history embeddings.
  Corresponds to ``l_X`` in [^1].
- `l_y::Integer`: The number of intervals in the source process to use in the history embeddings.
  Corresponds to ``l_Y`` in [^1].
- `auto_find_start_and_num_events::Bool = true`: When set to true, the start event will be set to
  the first event for which there are sufficient preceding events in all processes such that the
  embeddings can be constructed. The number of target events will be set such that all time between
  this first event and the last target event is included.
- `start_event::Integer = 1`: only used when `auto_find_start_and_num_events = false`. The index
  of the event in the target process from which to start the analysis.
- `num_target_events::Integer = length(target_events) - start_event`: only used when
  `auto_find_start_and_num_events = false`. The TE will be calculated on the time series from the
  timestamp of the `start_event`-th event of the target process to the timestamp of the
  `start_event + num_target_events`-th event of the target process.
- `num_samples_ratio::AbstractFloat = 1.0`: Controls the number of samples used to estimate the
  probability density of histories unconditional of the occurrence of events in the target process.
  This number of samples will be `num_samples_ratio * num_target_events`.
  Corresponds to ``N_U/N_X`` in [^1].
- `k_global::Integer = 5`: The number of nearest neighbours to consider in initial searches.
- `conditioning_events::Array{<:AbstractFloat} = [0.0]`: A list of the raw event times in the target
  process. Corresponds to ``Z_1`` in [^1].
  !!! info "Single conditioning process"
      Note that although the framework developed in [our paper](https://doi.org/10.1101/2020.06.16.154377)
      considers an arbitrary number of extra
      conditioning processes, at present the framework can only handle a single such process.
      This will change in future releases.
- `l_z::Integer = 0`: The number of intervals in the single conditioning process to use in the
  history embeddings. Corresponds to ``l_{Z_1}`` in [^1].
- `metric::Metric = Euclidean()`: The metric to use for nearest neighbour and radius searches.
- `is_surrogate::Bool = false`: If set to `true`, after the embeddings have been constructed, but
  before the TE is estimated, the source embeddings are permuted according to our local permutation
  scheme.
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
function calculate_TE_from_event_times(
    target_events::Array{<:AbstractFloat},
    source_events::Array{<:AbstractFloat},
    l_x::Integer,
    l_y::Integer;
    auto_find_start_and_num_events::Bool = true,
    num_target_events_cap::Integer = -1,
    start_event::Integer = 1,
    num_target_events::Integer = length(target_events) - start_event,
    num_samples_ratio::AbstractFloat = 1.0,
    k_global::Integer = 5,
    conditioning_events::Array{<:AbstractFloat} = [0.0],
    l_z::Integer = 0,
    metric::Metric = Euclidean(),
    AIS_only::Bool = false,
    kraskov_noise_level::AbstractFloat = 1e-8,
    is_surrogate::Bool = false,
    surrogate_num_samples_ratio::AbstractFloat = 1.0,
    k_perm::Integer = 5,
)

    preprocessed_data = CoTETE.preprocess_data(
        target_events,
        source_events,
        l_x,
        l_y,
        auto_find_start_and_num_events = auto_find_start_and_num_events,
        num_target_events_cap = num_target_events_cap,
        num_target_events = num_target_events,
        num_samples_ratio = num_samples_ratio,
        start_event = start_event,
        conditioning_events = conditioning_events,
        l_z = l_z,
        is_surrogate = is_surrogate,
        surrogate_num_samples_ratio = surrogate_num_samples_ratio,
        k_perm = k_perm,
    )

    TE = CoTETE.calculate_TE(
        preprocessed_data,
        k_global = k_global,
        metric = metric,
        AIS_only = AIS_only,
        kraskov_noise_level = kraskov_noise_level,
    )

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
    preprocessed_data = CoTETE.preprocess_data(
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

    TE = -calculate_TE(preprocessed_data, k_global = k_global, metric = metric, AIS_only = true)
    println(TE)

    surrogate_preprocessed_data = CoTETE.make_AIS_surrogate(
        target_events,
        preprocessed_data,
        Integer(round(
            surrogate_num_samples_ratio * length(preprocessed_data.representation_joint),
        )),
    )
    TE = -calculate_TE(surrogate_preprocessed_data, k_global = k_global, metric = metric, AIS_only = true)
    println(TE)
end

"""
    calculate_TE(
        preprocessed_data::PreprocessedData;
        k_global::Integer = 5,
        metric::Metric = Euclidean(),
    )
"""
function calculate_TE(
    preprocessed_data::PreprocessedData;
    k_global::Integer = 5,
    metric::Metric = Euclidean(),
    AIS_only::Bool = false,
    kraskov_noise_level::AbstractFloat = 1e-8,
)

    # Lets declare these to make the rest of this function less verbose
    l_x_plus_l_z = preprocessed_data.l_x + preprocessed_data.l_z
    l_y = preprocessed_data.l_y

    preprocessed_data.representation_joint[:, :] +=
        kraskov_noise_level .* randn(size(preprocessed_data.representation_joint))
    preprocessed_data.sampled_representation_joint[:, :] +=
        kraskov_noise_level .* randn(size(preprocessed_data.sampled_representation_joint))

    representation_conditionals = preprocessed_data.representation_joint[1:(l_x_plus_l_z), :]
    sampled_representation_conditionals =
        preprocessed_data.sampled_representation_joint[1:(l_x_plus_l_z), :]

    time =
        preprocessed_data.exclusion_windows[1, 2, end] -
        preprocessed_data.exclusion_windows[1, 1, 1]

    tree_joint =
        NearestNeighbors.KDTree(preprocessed_data.representation_joint, metric, reorder = false)

    tree_sampled_joint = NearestNeighbors.KDTree(
        preprocessed_data.sampled_representation_joint,
        metric,
        reorder = false,
    )

    tree_conditionals =
        NearestNeighbors.KDTree(representation_conditionals, metric, reorder = false)

    tree_sampled_conditionals =
        NearestNeighbors.KDTree(sampled_representation_conditionals, metric, reorder = false)

    TE = 0
    for i = 1:size(preprocessed_data.representation_joint, 2)


        indices_conditionals, radii_conditionals = NearestNeighbors.knn(
            tree_conditionals,
            representation_conditionals[:, i],
            preprocessed_data.exclusion_windows[:, :, i],
            preprocessed_data.exclusion_windows,
            k_global,
        )

        indices_sampled_conditionals, radii_sampled_conditionals = NearestNeighbors.knn(
            tree_sampled_conditionals,
            representation_conditionals[:, i],
            preprocessed_data.exclusion_windows[:, :, i],
            preprocessed_data.sampled_exclusion_windows,
            k_global,
        )

        radius_conditionals =
            max(maximum(radii_conditionals), maximum(radii_sampled_conditionals)) + 1e-6


        indices_conditionals = NearestNeighbors.inrange(
            tree_conditionals,
            representation_conditionals[:, i],
            preprocessed_data.exclusion_windows[:, :, i],
            preprocessed_data.exclusion_windows,
            radius_conditionals,
        )

        indices_sampled_conditionals = NearestNeighbors.inrange(
            tree_sampled_conditionals,
            representation_conditionals[:, i],
            preprocessed_data.exclusion_windows[:, :, i],
            preprocessed_data.sampled_exclusion_windows,
            radius_conditionals,
        )

        radius_conditionals = maximum(colwise(
            metric,
            representation_conditionals[:, i],
            representation_conditionals[:, indices_conditionals],
        ))

        radius_sampled_conditionals = maximum(colwise(
            metric,
            representation_conditionals[:, i],
            sampled_representation_conditionals[:, indices_sampled_conditionals],
        ))

        TE += (
            l_x_plus_l_z * log(radius_conditionals) -
            l_x_plus_l_z * log(radius_sampled_conditionals) -
            digamma(size(indices_conditionals)[1]) +
            digamma(size(indices_sampled_conditionals)[1])
        )

        if !AIS_only
            indices_joint, radii_joint = NearestNeighbors.knn(
                tree_joint,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.exclusion_windows,
                k_global,
            )

            indices_sampled_joint, radii_sampled_joint = NearestNeighbors.knn(
                tree_sampled_joint,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.sampled_exclusion_windows,
                k_global,
            )

            radius_joint = max(maximum(radii_joint), maximum(radii_sampled_joint)) + 1e-6

            indices_joint = NearestNeighbors.inrange(
                tree_joint,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.exclusion_windows,
                radius_joint,
            )

            indices_sampled_joint = NearestNeighbors.inrange(
                tree_sampled_joint,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.exclusion_windows[:, :, i],
                preprocessed_data.sampled_exclusion_windows,
                radius_joint,
            )

            radius_joint = maximum(colwise(
                metric,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.representation_joint[:, indices_joint],
            ))
            radius_sampled_joint = maximum(colwise(
                metric,
                preprocessed_data.representation_joint[:, i],
                preprocessed_data.sampled_representation_joint[:, indices_sampled_joint],
            ))

            TE += (
                -(l_x_plus_l_z + l_y)log(radius_joint) +
                (l_x_plus_l_z + l_y)log(radius_sampled_joint) +
                digamma(size(indices_joint)[1]) - digamma(size(indices_sampled_joint)[1])
            )
        end

    end

    return (TE / (time))

end



end
