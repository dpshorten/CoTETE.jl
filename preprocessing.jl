using Random: shuffle!, shuffle

include("NearestNeighbors.jl/src/NearestNeighbors.jl")

function make_one_embedding(
    time_point::AbstractFloat,
    event_time_arrays, #TODO: add type to this
    most_recent_event_indices::Array{<:Integer},
    embedding_lengths::Array{<:Integer},
)

    embedding = []
    candidate_start_times = []
    for i = 1:length(event_time_arrays)
        if embedding_lengths[i] > 0
            push!(embedding, time_point - event_time_arrays[i][most_recent_event_indices[i]])
            push!(candidate_start_times, event_time_arrays[i][most_recent_event_indices[i]])
        end
        for j = 2:embedding_lengths[i]
            push!(
                embedding,
                event_time_arrays[i][most_recent_event_indices[i]-j+2] -
                event_time_arrays[i][most_recent_event_indices[i]-j+1],
            )
            push!(candidate_start_times, event_time_arrays[i][most_recent_event_indices[i]-j+1])
        end
    end

    return embedding, minimum(candidate_start_times)

end

function make_embeddings_along_time_points(
    time_points::Array{<:AbstractFloat},
    start_time_point::Integer,
    num_time_points::Integer,
    event_time_arrays, #TODO: add type to this
    embedding_lengths::Array{<:Integer},
)

    trackers = ones(Integer, length(embedding_lengths))
    embeddings = []
    exclusion_windows = []
    for time_point in time_points[start_time_point:(start_time_point+num_time_points)]
        for i = 1:length(trackers)
            while (trackers[i] < length(event_time_arrays[i])) && (event_time_arrays[i][trackers[i]+1] < time_point)
                trackers[i] += 1
            end
        end
        #println(start_time_point, " ", trackers)
        embedding, start_time = make_one_embedding(time_point, event_time_arrays, trackers, embedding_lengths)
        push!(embeddings, embedding)
        push!(exclusion_windows, [start_time, time_point])
    end
    embeddings = hcat(embeddings...)

    exclusion_windows = hcat(exclusion_windows...)
    exclusion_windows = reshape(exclusion_windows, (1, size(exclusion_windows)...))

    return embeddings, exclusion_windows

end


function make_surrogate(
    representation_joint::Array{<:AbstractFloat},
    joint_exclusion_windows::Array{<:AbstractFloat},
    dense_sampled_representation_joint::Array{<:AbstractFloat},
    dense_sampled_joint_exclusion_windows::Array{<:AbstractFloat},
    metric::Metric,
    d_x_plus_d_c::Integer,
    k_perm::Integer,
)

    added_exclusion_windows = zeros(size(joint_exclusion_windows))

    tree = NearestNeighbors.KDTree(dense_sampled_representation_joint[1:d_x_plus_d_c, :], metric, reorder = false)

    new_joint = copy(representation_joint)
    permutation = shuffle(collect(1:size(new_joint, 2)))
    used_indices = zeros(size(new_joint, 2))
    for i = 1:size(permutation, 1)
        neighbour_indices, neighbour_radii = NearestNeighbors.knn(
            tree,
            new_joint[1:d_x_plus_d_c, permutation[i]],
            joint_exclusion_windows[:, :, permutation[i]],
            dense_sampled_joint_exclusion_windows,
            k_perm,
        )
        eligible_indices = neighbour_indices[findall(!in(used_indices), neighbour_indices)]
        if length(eligible_indices) > 0
            index = eligible_indices[rand(1:end)]
        else
            index = neighbour_indices[rand(1:end)]
        end
        used_indices[i] = index
        new_joint[(d_x_plus_d_c+1):end, permutation[i]] =
            dense_sampled_representation_joint[(d_x_plus_d_c+1):end, index]
        added_exclusion_windows[1, :, permutation[i]] = dense_sampled_joint_exclusion_windows[1, :, index]
    end

    new_joint_exclusion_windows = vcat(joint_exclusion_windows, added_exclusion_windows)
    #new_joint_exclusion_windows = joint_exclusion_windows

    return new_joint, new_joint_exclusion_windows

end

function construct_history_embeddings(
    target_events::Array{<:AbstractFloat},
    source_events::Array{<:AbstractFloat},
    d_x::Integer,
    d_y::Integer;
    start_event::Integer = min(d_x, d_y),
    num_target_events::Integer = length(target_events) - start_event,
    num_samples::Integer = num_target_events,
    noise_level::AbstractFloat = 1e-6,
    conditioning_events::Array{<:AbstractFloat} = [0.0],
    d_c::Integer = 0,
    is_surrogate::Bool = false,
    surrogate_upsample_ratio::AbstractFloat = 2.1,
    k_perm::Integer = 5,
    metric = Euclidean(),
)

    representation_joint, joint_exclusion_windows = make_embeddings_along_time_points(
        target_events,
        start_event,
        num_target_events,
        [target_events, conditioning_events, source_events],
        [d_x, d_c, d_y],
    )

    # sample_points = collect(
    #     joint_exclusion_windows[1, 1, 1]:((joint_exclusion_windows[
    #         1,
    #         2,
    #         end,
    #     ]-joint_exclusion_windows[1, 1, 1])/Float64(num_samples)):(joint_exclusion_windows[
    #         1,
    #         1,
    #         1,
    #     ]+((joint_exclusion_windows[1, 2, end]-joint_exclusion_windows[1, 1, 1])/2)),
    # )

    # sample_points = collect(
    #     joint_exclusion_windows[1, 1, 1]:((joint_exclusion_windows[
    #         1,
    #         2,
    #         end,
    #     ]-joint_exclusion_windows[1, 1, 1])/Float64(num_samples)):joint_exclusion_windows[1, 2, end],
    # )
    sample_points =
        joint_exclusion_windows[1, 2, 1] .+
        (joint_exclusion_windows[1, 2, end] - joint_exclusion_windows[1, 2, 1]) .* rand(num_samples)
    sort!(sample_points)

    sampled_representation_joint, sampled_joint_exclusion_windows = make_embeddings_along_time_points(
        sample_points,
        1,
        length(sample_points) - 2,
        [target_events, conditioning_events, source_events],
        [d_x, d_c, d_y],
    )

    if is_surrogate

        # dense_sample_points = collect(
        # joint_exclusion_windows[1, 1, 1]:((joint_exclusion_windows[
        #     1,
        #     2,
        #     end,
        # ]-joint_exclusion_windows[1, 1, 1])/Float64(surrogate_upsample_ratio * num_samples)):joint_exclusion_windows[1, 2, end],
        # )
        dense_sample_points =
            joint_exclusion_windows[1, 2, 1] .+
            (joint_exclusion_windows[1, 2, end] - joint_exclusion_windows[1, 2, 1]) .*
            rand(Int(round(surrogate_upsample_ratio * num_samples)))
        sort!(dense_sample_points)
        dense_sampled_representation_joint, dense_sampled_joint_exclusion_windows = make_embeddings_along_time_points(
            dense_sample_points,
            1,
            length(sample_points) - 2,
            [target_events, conditioning_events, source_events],
            [d_x, d_c, d_y],
        )

        representation_joint, joint_exclusion_windows = make_surrogate(
            representation_joint,
            joint_exclusion_windows,
            dense_sampled_representation_joint,
            dense_sampled_joint_exclusion_windows,
            metric,
            d_x + d_c,
            k_perm,
        )

        # added_exclusion_windows_sampled = zeros(size(sampled_joint_exclusion_windows))
        # permutation =  shuffle(collect(1:size(sampled_joint_exclusion_windows, 3))
        # added_exclusion_windows_sampled[1, :, permutation] = sampled_joint_exclusion_windows[1, :, collect(1:size(sampled_joint_exclusion_windows, 3)]
        # sampled_joint_exclusion_windows = vcat(sampled_joint_exclusion_windows, added_exclusion_windows_sampled)
    end


    sampled_representation_joint += noise_level .* randn(size(sampled_representation_joint))

    representation_conditionals = representation_joint[1:(d_x+d_c), :]
    sampled_representation_conditionals = sampled_representation_joint[1:(d_x+d_c), :]

    return (
        representation_joint,
        joint_exclusion_windows,
        representation_conditionals,
        sampled_representation_joint,
        sampled_joint_exclusion_windows,
        sampled_representation_conditionals,
    )

end
