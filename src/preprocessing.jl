using Random: shuffle!, shuffle

push!(LOAD_PATH, "NearestNeighbors.jl/src/NearestNeighbors.jl")
include("NearestNeighbors.jl/src/NearestNeighbors.jl")

"""
    representation_joint::Array{<:AbstractFloat, 2}
    representation_conditionals::Array{<:AbstractFloat, 2}
    exclusion_windows::Array{<:AbstractFloat, 3}
    sampled_representation_joint::Array{<:AbstractFloat, 2}
    sampled_representation_conditionals::Array{<:AbstractFloat, 2}
    sampled_exclusion_windows::Array{<:AbstractFloat, 3}

The transformed data that is fed into the search trees.

- `representation_joint::Array{<:AbstractFloat, 2}`: Contains the history representation of the source, target and
  extra conditioning processes at each target event. Has dimension ``(l_X + l_Y + l_{Z_1}) \\times N_X``. Rows 1 to
  ``l_X`` (inclusive) contain the components relating to the target process. Rows ``l_X + 1`` to ``l_X + l_Y`` contain the
  components relating to the source process. Rows ``l_X l_Y + 1`` to ``l_X + l_Y + l_{Z_1}`` contain the
  components relating to the conditioning process. A similar convention applies to all the other representation variables.
- `representation_conditionals::Array{<:AbstractFloat, 2}`: Contains the history representation of the target and
  extra conditioning processes at each target event. Has dimension ``(l_X + l_{Z_1}) \\times N_X``.
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
  extra conditioning processes at each sample point. Has dimension ``(l_X + l_Y + l_{Z_1}) \\times N_U``.
- `sampled_representation_conditionals::Array{<:AbstractFloat, 2}`: Contains the history representation of the target and
  extra conditioning processes at each sample point. Has dimension ``(l_X + l_{Z_1}) \\times N_U``.
- `sampled_exclusion_windows::Array{<:AbstractFloat, 3}`: Same as for the `exclusion_windows`, but contains the windows
  around the representations constructed at sample points.
"""
struct PreprocessedData
    representation_joint::Array{<:AbstractFloat,2}
    exclusion_windows::Array{<:AbstractFloat,3}
    sampled_representation_joint::Array{<:AbstractFloat,2}
    sampled_exclusion_windows::Array{<:AbstractFloat,3}
    l_x::Integer
    l_y::Integer
    l_z::Integer
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

```jldoctest calculate_TE_from_event_times
julia> source = cumsum(ones(20)) .- 0.5; # source is {0.5, 1.5, 2.5, ...}

julia> conditional = cumsum(ones(20)) .- 0.25; # conditional is {0.75, 1.75, 2.75, ...}

julia> target = cumsum(ones(20)); # target is {1, 2, 3, ...}

julia> using CoTETE

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
            push!(embedding, observation_time_point - event_time_arrays[i][most_recent_event_indices[i]])
            push!(candidate_start_times, event_time_arrays[i][most_recent_event_indices[i]])
        end
        # Embed the subsequent inter-event intervals
        for j = 2:embedding_lengths[i]
            push!(
                embedding,
                event_time_arrays[i][most_recent_event_indices[i]-j+2] -
                event_time_arrays[i][most_recent_event_indices[i]-j+1],
            )
            # Keep track of the raw event times so that the first can be found later.
            push!(candidate_start_times, event_time_arrays[i][most_recent_event_indices[i]-j+1])
        end
    end

    return embedding, minimum(candidate_start_times)

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

```jldoctest calculate_TE_from_event_times
julia> source = cumsum(ones(20)) .- 0.5; # source is {0.5, 1.5, 2.5, ...}

julia> conditional = cumsum(ones(20)) .- 0.25; # conditional is {0.75, 1.75, 2.75, ...}

julia> target = cumsum(ones(20)); # target is {1, 2, 3, ...}

julia> observation_points = cumsum(ones(20)) .- 0.75; # observation points are {0.25, 1.25, 2.25, ...}

julia> using CoTETE

julia> CoTETE.make_embeddings_along_observation_time_points(observation_points, 5, 2, [target, source, conditional], [2, 1, 1])
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
        observation_time_points[start_observation_time_point:(start_observation_time_point+num_observation_time_points_to_use)]
        # Update the position of each tracker variable
        for i = 1:length(trackers)
            while (trackers[i] < length(event_time_arrays[i])) &&
                  (event_time_arrays[i][trackers[i]+1] < observation_time_point)
                trackers[i] += 1
            end
        end
        embedding, start_time =
            make_one_embedding(observation_time_point, event_time_arrays, trackers, embedding_lengths)
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
    function make_surrogate(
        representation_joint::Array{<:AbstractFloat},
        exclusion_windows::Array{<:AbstractFloat},
        dense_sampled_representation_joint::Array{<:AbstractFloat},
        dense_sampled_exclusion_windows::Array{<:AbstractFloat},
        metric::Metric,
        l_x_plus_l_z::Integer,
        k_perm::Integer,
    )
"""
function make_surrogate(
    preprocessed_data::PreprocessedData,
    dense_sampled_representation_joint::Array{<:AbstractFloat},
    dense_sampled_exclusion_windows::Array{<:AbstractFloat},
    metric::Metric,
    l_x_plus_l_z::Integer,
    k_perm::Integer,
)

    added_exclusion_windows = zeros(size(exclusion_windows))

    tree = NearestNeighbors.KDTree(dense_sampled_representation_joint[1:l_x_plus_l_z, :], metric, reorder = false)

    new_joint = copy(preprocessed_data.representation_joint)
    permutation = shuffle(collect(1:size(new_joint, 2)))
    used_indices = zeros(size(new_joint, 2))
    for i = 1:size(permutation, 1)
        neighbour_indices, neighbour_radii = NearestNeighbors.knn(
            tree,
            new_joint[1:l_x_plus_l_z, permutation[i]],
            preprocessed_data.exclusion_windows[:, :, permutation[i]],
            dense_sampled_exclusion_windows,
            k_perm,
        )
        eligible_indices = neighbour_indices[findall(!in(used_indices), neighbour_indices)]
        if length(eligible_indices) > 0
            index = eligible_indices[rand(1:end)]
        else
            index = neighbour_indices[rand(1:end)]
        end
        used_indices[i] = index
        new_joint[(l_x_plus_l_z+1):end, permutation[i]] =
            dense_sampled_representation_joint[(l_x_plus_l_z+1):end, index]
        added_exclusion_windows[1, :, permutation[i]] = dense_sampled_exclusion_windows[1, :, index]
    end

    new_exclusion_windows = vcat(exclusion_windows, added_exclusion_windows)

    return new_joint, new_exclusion_windows
end


"""
    function preprocess_data(
        target_events::Array{<:AbstractFloat},
        source_events::Array{<:AbstractFloat},
        l_x::Integer,
        l_y::Integer;
        auto_find_start_and_num_events::Bool = true,
        start_event::Integer = 1,
        num_target_events::Integer = length(target_events) - start_event,
        num_samples_ratio::AbstractFloat = 1.0,
        noise_level::AbstractFloat = 1e-8,
        conditioning_events::Array{<:AbstractFloat} = [0.0],
        l_z::Integer = 0,
        is_surrogate::Bool = false,
        surrogate_num_samples_ratio::AbstractFloat = 1.0,
        k_perm::Integer = 5,
        metric = Euclidean(),
    )
"""
function preprocess_data(
    target_events::Array{<:AbstractFloat},
    source_events::Array{<:AbstractFloat},
    l_x::Integer,
    l_y::Integer;
    auto_find_start_and_num_events::Bool = true,
    start_event::Integer = 1,
    num_target_events::Integer = length(target_events) - start_event,
    num_target_events_cap::Integer = -1,
    num_samples_ratio::AbstractFloat = 1.0,
    noise_level::AbstractFloat = 1e-8,
    conditioning_events::Array{<:AbstractFloat} = [0.0],
    l_z::Integer = 0,
    is_surrogate::Bool = false,
    surrogate_num_samples_ratio::AbstractFloat = 1.0,
    k_perm::Integer = 5,
    metric = Euclidean(),
)


    if auto_find_start_and_num_events
        # This will ensure that we have at least enough events to make the target embedding
        start_event = l_x + 1
        while source_events[l_y] > target_events[start_event]
            start_event += 1
        end
        num_target_events = length(target_events) - start_event
        if num_target_events_cap > 0 && num_target_events > num_target_events_cap
            num_target_events = num_target_events_cap
        end
    end

    num_samples = Int(round(num_samples_ratio * num_target_events))

    representation_joint, exclusion_windows = make_embeddings_along_observation_time_points(
        target_events,
        start_event,
        num_target_events,
        [target_events, conditioning_events, source_events],
        [l_x, l_z, l_y],
    )


    sample_points =
        exclusion_windows[1, 2, 1] .+ (exclusion_windows[1, 2, end] - exclusion_windows[1, 2, 1]) .* rand(num_samples)
    sort!(sample_points)

    sampled_representation_joint, sampled_exclusion_windows = make_embeddings_along_observation_time_points(
        sample_points,
        1,
        length(sample_points) - 2,
        [target_events, conditioning_events, source_events],
        [l_x, l_z, l_y],
    )

    if is_surrogate
        surrogate_num_samples = Int(round(surrogate_num_samples_ratio * num_target_events))
        dense_sample_points =
            exclusion_windows[1, 2, 1] .+
            (exclusion_windows[1, 2, end] - exclusion_windows[1, 2, 1]) .* rand(surrogate_num_samples)
        sort!(dense_sample_points)
        dense_sampled_representation_joint, dense_sampled_exclusion_windows =
            make_embeddings_along_observation_time_points(
                dense_sample_points,
                1,
                length(sample_points) - 2,
                [target_events, conditioning_events, source_events],
                [l_x, l_z, l_y],
            )

        representation_joint, exclusion_windows = make_surrogate(
            representation_joint,
            exclusion_windows,
            dense_sampled_representation_joint,
            dense_sampled_exclusion_windows,
            metric,
            l_x + l_z,
            k_perm,
        )

    end

    #representation_joint += noise_level .* randn(size(representation_joint))
    #sampled_representation_joint += noise_level .* randn(size(sampled_representation_joint))

    #representation_conditionals = representation_joint[1:(l_x+l_z), :]
    #sampled_representation_conditionals = sampled_representation_joint[1:(l_x+l_z), :]

    return PreprocessedData(
        representation_joint,
        exclusion_windows,
        sampled_representation_joint,
        sampled_exclusion_windows,
        l_x,
        l_y,
        l_z
    )

end
