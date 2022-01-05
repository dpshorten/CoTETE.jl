using Random: shuffle!, shuffle
using StatsBase: ordinalrank

push!(LOAD_PATH, "NearestNeighbors.jl/src/NearestNeighbors.jl")
include("NearestNeighbors.jl/src/NearestNeighbors.jl")

"""
    representation_joint::Array{<:AbstractFloat, 2}
    exclusion_windows::Array{<:AbstractFloat, 3}
    sampled_representation_joint::Array{<:AbstractFloat, 2}
    sampled_exclusion_windows::Array{<:AbstractFloat, 3}
    start_timestamp::AbstractFloat
    end_timestamp::AbstractFloat

The transformed data that is fed into the search trees.

- `representation_joint::Array{<:AbstractFloat, 2}`: Contains the history representation of the source, target and
  extra conditioning process at each target event. Has dimension ``(l_X + l_{Z_1} + l_Y) \\times N_X``. Rows 1 to
  ``l_X`` (inclusive) contain the components relating to the target process. Rows ``l_X + 1`` to ``l_X + l_{Z_1}`` contain the
  components relating to the conditioning process. Rows ``l_X  + l_{Z_1} + 1`` to ``l_X + l_{Z_1} + l_Y`` contain the
  components relating to the source process. A similar convention is used by `sampled_representation_joint`.
  Note that we do not include an array in this struct to keep track of the history embeddings for the conditioning variables.
  This is because this array is simply the first ``l_X + l_{Z_1}`` rows of this array and so can easily be constructed on the
  fly later.
- `exclusion_windows::Array{<:AbstractFloat, 3}`: Contains records of the time windows around each representation made at
  target events which must be excluded when doing ``k``NN searches from that representation. By default, each representation has
  the window that is bound to the left by the first point in
  time that was used to make an embedding at that sample and to the right by the timestamp of the target event itself.
  An extra window might be added if the representation is a surrogate. In this case, the second window will be the original
  window of the representation with which the source component is swapped. Has dimension ``N_E \\times 2 \\times N_X``.
  ``N_E`` is the number of exclusion windows the representation has (one by default, two if it is a surrogate).
  Note that a single set of exclusion windows is used for the representations of both the joints and the conditionals.
  Using separate sets of windows would allow them to be slightly smaller, but the effect will be negligible for longer
  processes.
- `sampled_representation_joint::Array{<:AbstractFloat, 2}`: Contains the history representation of the source, target and
  extra conditioning processes at each sample point. Has dimension ``(l_X + l_Y + l_{Z_1}) \\times N_U``. See the description of
  `representation_joint` for a description of how the variables are split accross the dimensions.
- `sampled_exclusion_windows::Array{<:AbstractFloat, 3}`: Same as for the `exclusion_windows`, but contains the windows
  around the representations constructed at sample points.
- `start_timestamp::AbstractFloat`: The raw timestamp of the first target event that is included in the analysis.
- `end_timestamp::AbstractFloat`: The raw timestamp of the last target event that is included in the analysis.
"""
mutable struct PreprocessedData
    representation_joint::Array{<:AbstractFloat,2}
    exclusion_windows::Array{<:AbstractFloat,3}
    sampled_representation_joint::Array{<:AbstractFloat,2}
    sampled_exclusion_windows::Array{<:AbstractFloat,3}
    empirical_cdf::Array{<:AbstractFloat,2}
    start_timestamp::AbstractFloat
    end_timestamp::AbstractFloat
end

"""
    function make_one_embedding(
        observation_time_point::AbstractFloat,
        event_time_arrays::Array{<:Array{<:AbstractFloat, 1}, 1},
        most_recent_event_indices::Array{<:Integer},
        embedding_lengths::Array{<:Integer},
    )

Constructs the history embedding from a given point in time. Also returns the timestamp of the earliest event
used in the construction of the embedding. This is used for recording the exclusion windows.

# Example

```jldoctest
julia> source = cumsum(ones(20)) .- 0.5; # source is {0.5, 1.5, 2.5, ...}

julia> conditional = cumsum(ones(20)) .- 0.25; # conditional is {0.75, 1.75, 2.75, ...}

julia> target = cumsum(ones(20)); # target is {1, 2, 3, ...}

julia> CoTETE.make_one_embedding(5.25, [target, source, conditional], [5, 5, 5], [2, 3, 1])
(Any[0.25, 1.0, 0.75, 1.0, 1.0, 0.5], 2.5)
```
"""
function make_one_embedding(
    observation_time_point::AbstractFloat,
    event_time_arrays::Array{<:Array{<:AbstractFloat,1},1},
    most_recent_event_indices::Array{<:Integer},
    embedding_lengths::Array{<:Integer},
)
    embedding = []
    candidate_start_times = []
    # Loop over the set of processes ({X, Y} or {X, Y, Z_1})
    for i = 1:length(event_time_arrays)
        # Embed the interval from the observation point to the most recent event (if necessary)
        if embedding_lengths[i] > 0
            push!(
                embedding,
                observation_time_point - event_time_arrays[i][most_recent_event_indices[i]],
            )
            push!(candidate_start_times, event_time_arrays[i][most_recent_event_indices[i]])
        end
        # Embed the subsequent inter-event intervals
        for j = 2:embedding_lengths[i]
            push!(
                embedding,
                #observation_time_point -
                event_time_arrays[i][most_recent_event_indices[i]-j+2] -
                event_time_arrays[i][most_recent_event_indices[i]-j+1],
            )
            # Keep track of the raw event times so that the first can be found later.
            push!(candidate_start_times, event_time_arrays[i][most_recent_event_indices[i]-j+1])
        end
    end

    return embedding, (minimum(candidate_start_times))

end

"""
    function make_embeddings_along_observation_time_points(
        observation_time_points::Array{<:AbstractFloat},
        start_observation_time_point::Integer,
        num_observation_time_points_to_use::Integer,
        event_time_arrays::Array{<:Array{<:AbstractFloat,1},1},
        embedding_lengths::Array{<:Integer},
    )

Constructs a set of embeddings from a set of observation points. The observation points and the raw event times are
assumed to be sorted. Also returns the exlcusion windows.

# Example

```jldoctest
julia> source = cumsum(ones(20)) .- 0.5; # source is {0.5, 1.5, 2.5, ...}

julia> conditional = cumsum(ones(20)) .- 0.25; # conditional is {0.75, 1.75, 2.75, ...}

julia> target = cumsum(ones(20)); # target is {1, 2, 3, ...}

julia> observation_points = cumsum(ones(20)) .- 0.75; # observation points are {0.25, 1.25, 2.25, ...}

julia> CoTETE.make_embeddings_along_observation_time_points(observation_points, 5, 3, [target, source, conditional], [2, 1, 1])
([0.25 0.25 0.25; 1.0 1.0 1.0; 0.75 0.75 0.75; 0.5 0.5 0.5], [3.0 4.25]

[4.0 5.25]

[5.0 6.25])

```
"""
function make_embeddings_along_observation_time_points(
    observation_time_points::Array{<:AbstractFloat},
    start_observation_time_point::Integer,
    num_observation_time_points_to_use::Integer,
    event_time_arrays::Array{<:Array{<:AbstractFloat,1},1},
    embedding_lengths::Array{<:Integer},
)
    # Variables that track the index of the most recent event in each process
    trackers = ones(Integer, length(embedding_lengths))
    embeddings = []
    exclusion_windows = []
    for observation_time_point in
        observation_time_points[start_observation_time_point:(start_observation_time_point+num_observation_time_points_to_use-1)]
        # Update the position of each tracker variable
        for i = 1:length(trackers)
            while (trackers[i] < length(event_time_arrays[i])) &&
                (event_time_arrays[i][trackers[i]+1] < observation_time_point)
                trackers[i] += 1
            end
        end
        embedding, start_time = make_one_embedding(
            observation_time_point,
            event_time_arrays,
            trackers,
            embedding_lengths,
        )
        push!(embeddings, embedding)
        push!(exclusion_windows, [start_time, observation_time_point])
    end
    # Conver the embeddings from an array of arrays to a 2d array.
    embeddings = hcat(embeddings...)
    # Do the same conversion on the exclusion windows
    exclusion_windows = hcat(exclusion_windows...)
    # Add an extra dimension to the exclusion windows. This allows more windows to be added later. i.e. when making surrogates.
    exclusion_windows = reshape(exclusion_windows, (1, size(exclusion_windows)...))

    return embeddings, exclusion_windows

end

"""
    function transform_marginals_to_uniform!(preprocessed_data::PreprocessedData)

    Independently transforms each dimension of the history embeddings to be uniformly distributed.
"""
function transform_marginals_to_uniform!(preprocessed_data::PreprocessedData)

    combined =
        hcat(preprocessed_data.representation_joint, preprocessed_data.sampled_representation_joint)
    preprocessed_data.empirical_cdf = zeros(size(combined))
    for dim = 1:size(combined, 1)
        preprocessed_data.empirical_cdf[dim, :] = sort(combined[dim, :])
    end

    transform_marginals_to_uniform_on_cdf!(
        preprocessed_data,
        preprocessed_data.representation_joint,
    )

    transform_marginals_to_uniform_on_cdf!(
        preprocessed_data,
        preprocessed_data.sampled_representation_joint,
    )
end

"""
    function transform_marginals_to_uniform_on_cdf!(
        preprocessed_data::PreprocessedData,
        history_embeddings::Array{<:AbstractFloat,2},
    )

    Helper method reused for transforming both the embeddings sampled at events and those sampled
    randomly.
"""
function transform_marginals_to_uniform_on_cdf!(
    preprocessed_data::PreprocessedData,
    history_embeddings::Array{<:AbstractFloat,2},
)
    for dim = 1:size(history_embeddings, 1)
        for i = 1:size(history_embeddings, 2)
            upper_index = searchsortedfirst(
                preprocessed_data.empirical_cdf[dim, :],
                history_embeddings[dim, i],
            )
            if upper_index == size(preprocessed_data.empirical_cdf, 2) + 1
                history_embeddings[dim, i] = 1.0
            elseif upper_index == 1
                history_embeddings[dim, i] = 1 / size(preprocessed_data.empirical_cdf, 2)
            else
                history_embeddings[dim, i] =
                    (
                        upper_index - 1 + (
                            (
                                history_embeddings[dim, i] -
                                preprocessed_data.empirical_cdf[dim, upper_index-1]
                            ) / (
                                preprocessed_data.empirical_cdf[dim, upper_index] -
                                preprocessed_data.empirical_cdf[dim, upper_index-1]
                            )
                        )
                    ) / size(preprocessed_data.empirical_cdf, 2)
            end
        end
    end
end

"""
    function make_surrogate!(
        parameters::CoTETEParameters,
        preprocessed_data::PreprocessedData,
        target_events::Array{<:AbstractFloat},
        source_events::Array{<:AbstractFloat};
        conditioning_events::Array{<:AbstractFloat} = Float32[],
    )

Edit the source component of `preprocessed_data.representation_joint` such that it conforms to the
null hypothesis of conditional independence.
"""
function make_surrogate!(
    parameters::CoTETEParameters,
    preprocessed_data::PreprocessedData,
    target_events::Array{<:AbstractFloat},
    source_events::Array{<:AbstractFloat};
    conditioning_events::Array{<:Array{<:AbstractFloat,1},1} = Float32[],
    only_dummy_exclusion_windows::Bool = false,
)

    # Declare this to make the code slightly less verbose
    l_x_plus_l_z = parameters.l_x + sum(parameters.l_z)

    # Construct a new sampling of the distribution P_U
    num_sample_points = Int(round(
        parameters.surrogate_num_samples_ratio * size(preprocessed_data.representation_joint, 2),
    ))

    sample_points = construct_sample_points_array(
        parameters,
        num_sample_points,
        preprocessed_data.start_timestamp,
        preprocessed_data.end_timestamp,
        target_events,
    )

    array_of_event_arrays = [target_events]
    array_of_dimensions = [parameters.l_x]
    for i = 1:length(parameters.l_z)
        push!(array_of_event_arrays, conditioning_events[i])
        push!(array_of_dimensions, parameters.l_z[i])
    end
    push!(array_of_event_arrays, source_events)
    push!(array_of_dimensions, parameters.l_y)

    resampled_representation_joint, resampled_exclusion_windows =
        make_embeddings_along_observation_time_points(
            sample_points,
            1,
            length(sample_points) - 2, #TODO Come back and look at this -2
            array_of_event_arrays,
            array_of_dimensions,
        )

    if parameters.transform_to_uniform
        transform_marginals_to_uniform_on_cdf!(preprocessed_data, resampled_representation_joint)
    end

    tree = NearestNeighbors.KDTree(
        resampled_representation_joint[1:l_x_plus_l_z, :],
        parameters.metric,
        reorder = false,
    )

    added_exclusion_windows = zeros(size(preprocessed_data.exclusion_windows))
    permutation = shuffle(collect(1:size(preprocessed_data.representation_joint, 2)))
    used_indices = zeros(size(preprocessed_data.representation_joint, 2))
    for i = 1:length(permutation)
        neighbour_indices, neighbour_radii = NearestNeighbors.knn(
            tree,
            preprocessed_data.representation_joint[1:l_x_plus_l_z, permutation[i]],
            preprocessed_data.exclusion_windows[:, :, permutation[i]],
            resampled_exclusion_windows,
            parameters.k_perm,
        )
        # Find the neighbouring indices that have not already been used
        eligible_indices = neighbour_indices[findall(!in(used_indices), neighbour_indices)]
        if length(eligible_indices) > 0
            index = eligible_indices[rand(1:end)]
        else
            index = neighbour_indices[rand(1:end)]
        end
        used_indices[i] = index
        if !only_dummy_exclusion_windows
            preprocessed_data.representation_joint[(l_x_plus_l_z+1):end, permutation[i]] =
                resampled_representation_joint[(l_x_plus_l_z+1):end, index]
        end
        added_exclusion_windows[1, :, permutation[i]] = resampled_exclusion_windows[1, :, index]
    end

    preprocessed_data.exclusion_windows =
        vcat(preprocessed_data.exclusion_windows, added_exclusion_windows)
end

"""
    function make_AIS_surrogate!(
        parameters::CoTETEParameters,
        preprocessed_data::PreprocessedData,
        target_events::Array{<:AbstractFloat},
    )

Edit `preprocessed_data.representation_joint` such that it conforms to the
null hypothesis of the target histories being independent of events in the target process.
"""
function make_AIS_surrogate!(
    parameters::CoTETEParameters,
    preprocessed_data::PreprocessedData,
    target_events::Array{<:AbstractFloat},
)
    num_resample_points = min(
        Int(round(
            parameters.surrogate_num_samples_ratio *
            size(preprocessed_data.representation_joint, 2),
        )),
        size(preprocessed_data.representation_joint, 2),
    )
    resample_points = construct_sample_points_array(
        parameters,
        num_resample_points,
        preprocessed_data.start_timestamp,
        preprocessed_data.end_timestamp,
        target_events,
    )

    resampled_representation_joint, resampled_exclusion_windows =
        make_embeddings_along_observation_time_points(
            resample_points,
            1,
            length(resample_points),
            [target_events, Float64[], Float64[]],
            [parameters.l_x, 0, 0],
        )
    shuffled_indices_of_resample = shuffle(collect(1:size(resampled_representation_joint, 2)))
    for i = 1:size(preprocessed_data.representation_joint, 2)
        preprocessed_data.representation_joint[:, i] =
            resampled_representation_joint[:, shuffled_indices_of_resample[i]]
        preprocessed_data.exclusion_windows[:, :, i] =
            resampled_exclusion_windows[:, :, shuffled_indices_of_resample[i]]
    end
end

"""
    function construct_sample_points_array(
        parameters::CoTETEParameters,
        num_samples::Integer,
        start_timestamp::AbstractFloat,
        end_timestamp::AbstractFloat,
        target_events::Array{<:AbstractFloat}
    )

    Constructs the array of random sample points according to the chosen method.
"""
function construct_sample_points_array(
    parameters::CoTETEParameters,
    num_samples::Integer,
    start_timestamp::AbstractFloat,
    end_timestamp::AbstractFloat,
    target_events::Array{<:AbstractFloat},
)

    if !(parameters.sampling_method in ["random_uniform", "fixed_interval", "jittered_target"])
        error("sampling_method must be one of: \"random_uniform\", \"fixed_interval\" or \"jittered_target\"")
    elseif parameters.sampling_method == "random_uniform"
        sample_points = start_timestamp .+ ((end_timestamp - start_timestamp) * rand(num_samples))
        sort!(sample_points)
    elseif parameters.sampling_method == "fixed_interval"
        sample_points = collect(range(start_timestamp, stop = end_timestamp, length = num_samples))
    else
        sample_points = []
        temp_target_events = deepcopy(target_events)
        temp_target_events = temp_target_events[temp_target_events.>=start_timestamp]
        temp_target_events = temp_target_events[temp_target_events.<=end_timestamp]
        for i = 1:Int(parameters.surrogate_num_samples_ratio)
            append!(sample_points, temp_target_events)
        end
        sample_points =
            sample_points +
            parameters.jittered_sampling_noise .* (rand(length(sample_points)) .- 0.5)
        sort!(sample_points)
        for i = 1:length(sample_points)
            if sample_points[i] < start_timestamp || sample_points[i] > end_timestamp
                sample_points[i] = start_timestamp + ((end_timestamp - start_timestamp) * rand())
            else
                break
            end
        end
    end

    return sample_points
end

"""
    function preprocess_event_times(
        parameters::CoTETEParameters,
        target_events::Array{<:AbstractFloat};
        source_events::Array{<:AbstractFloat} = Float32[],
        conditioning_events::Array{<:AbstractFloat} = Float32[]
    )

Use the raw event times to create the history embeddings and other prerequisites for estimating the TE.

```jldoctest
julia> parameters = CoTETE.CoTETEParameters(l_x = 1, l_y = 1);

julia> source = cumsum(ones(5)) .- 0.5; # source is {0.5, 1.5, 2.5, ...}

julia> target = cumsum(ones(5)); # target is {1, 2, 3, ...}

julia> preprocessed_data = CoTETE.preprocess_event_times(parameters, target, source_events = source);

julia> println(preprocessed_data.representation_joint) # All target events will be one unit back, all source events 0.5 units
[1.0 1.0 1.0 1.0; 0.5 0.5 0.5 0.5]

```
"""
function preprocess_event_times(
    parameters::CoTETEParameters,
    target_events::Array{<:AbstractFloat};
    source_events::Array{<:AbstractFloat} = Float32[],
    conditioning_events::Array{<:Array{<:AbstractFloat,1},1} = [Float32[]],
)

    # We first need to figure out which target event will be the first and how many we will include
    # in the analysis.
    if parameters.auto_find_start_and_num_events
        #=
          When auto-finding we will use the first target event that has sufficient events preceding it
          in all processes such that the history embeddings can be built.
        =#
        index_of_target_start_event = parameters.l_x + 1
        if parameters.l_y > 0
            while source_events[parameters.l_y] > target_events[index_of_target_start_event]
                index_of_target_start_event += 1
            end
        end
        if length(parameters.l_z) > 0
            for i = 1:length(parameters.l_z)
                while conditioning_events[i][parameters.l_z[i]] >
                      target_events[index_of_target_start_event]
                    index_of_target_start_event += 1
                end
            end
        end
        num_target_events = length(target_events) - index_of_target_start_event + 1
        if parameters.num_target_events_cap > 0 &&
           num_target_events > parameters.num_target_events_cap
            num_target_events = parameters.num_target_events_cap
        end
    else
        # Use the user specified values
        num_target_events = parameters.num_target_events
        index_of_target_start_event = parameters.start_event
    end

    start_timestamp = target_events[index_of_target_start_event]
    end_timestamp = target_events[index_of_target_start_event+num_target_events-1]

    array_of_event_arrays = [target_events]
    array_of_dimensions = [parameters.l_x]
    for i = 1:length(parameters.l_z)
        push!(array_of_event_arrays, conditioning_events[i])
        push!(array_of_dimensions, parameters.l_z[i])
    end
    push!(array_of_event_arrays, source_events)
    push!(array_of_dimensions, parameters.l_y)

    representation_joint, exclusion_windows = make_embeddings_along_observation_time_points(
        target_events,
        index_of_target_start_event,
        num_target_events,
        array_of_event_arrays,
        array_of_dimensions,
    )

    num_samples = Int(round(parameters.num_samples_ratio * num_target_events))

    sample_points = construct_sample_points_array(
        parameters,
        num_samples,
        start_timestamp,
        end_timestamp,
        target_events,
    )

    sampled_representation_joint, sampled_exclusion_windows =
        make_embeddings_along_observation_time_points(
            sample_points,
            1,
            length(sample_points),
            array_of_event_arrays,
            array_of_dimensions,
        )

    preprocessed_data = PreprocessedData(
        representation_joint,
        exclusion_windows,
        sampled_representation_joint,
        sampled_exclusion_windows,
        [0.0 0.0; 0.0 0.0],
        start_timestamp,
        end_timestamp,
    )

    if parameters.transform_to_uniform
        transform_marginals_to_uniform!(preprocessed_data)
    end

    return preprocessed_data
end
