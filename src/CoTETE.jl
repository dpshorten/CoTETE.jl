module CoTETE

push!(LOAD_PATH,"NearestNeighbors.jl/src/NearestNeighbors.jl")


using Distances: evaluate, colwise, Metric, Chebyshev, Euclidean
using SpecialFunctions: digamma, gamma

include("preprocessing.jl")

"""
    function calculate_TE_from_event_times(
        target_events::Array{<:AbstractFloat},
        source_events::Array{<:AbstractFloat},
        l_x::Integer,
        l_y::Integer;
        start_event::Integer
        num_target_events::Integer = length(target_events) - start_event,
        num_samples::Integer = num_target_events,
        k::Integer = 5,
        conditioning_events::Array{<:AbstractFloat} = [0.0],
        l_z::Integer = 0,
        metric::Metric = Euclidean(),
        is_surrogate::Bool = false,
        surrogate_upsample_ratio::AbstractFloat = 2.1,
        k_perm::Integer = 5,
        )

    Estimates the TE from lists of raw event times.

    # Arguments
    - `target_events::Array{<:AbstractFloat}`: A list of the raw event times in the target process.
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
    k::Integer = 5,
    conditioning_events::Array{<:AbstractFloat} = [0.0],
    l_z::Integer = 0,
    metric::Metric = Euclidean(),
    is_surrogate::Bool = false,
    surrogate_num_samples_ratio::AbstractFloat = 1.0,
    k_perm::Integer = 5,
)

    representation_joint,
    joint_exclusion_windows,
    representation_conditionals,
    sampled_representation_joint,
    sampled_joint_exclusion_windows,
    sampled_representation_conditionals, = CoTETE.construct_history_embeddings(
        target_events,
        source_events,
        l_x,
        l_y,
        auto_find_start_and_num_events = auto_find_start_and_num_events,
        num_target_events = num_target_events,
        num_samples_ratio = num_samples_ratio,
        start_event = start_event,
        conditioning_events = conditioning_events,
        l_z = l_z,
        is_surrogate = is_surrogate,
        surrogate_num_samples_ratio = surrogate_num_samples_ratio,
        k_perm = k_perm,
        metric = metric,
    )

    TE = CoTETE.calculate_TE(
        representation_joint,
        representation_conditionals,
        sampled_representation_joint,
        sampled_representation_conditionals,
        joint_exclusion_windows,
        sampled_joint_exclusion_windows,
        k = k,
        metric = metric,
    )

    return TE

end

"""
    foo()

    Returns the TE.
"""
function calculate_TE(
    representation_joint::Array{<:AbstractFloat},
    representation_conditionals::Array{<:AbstractFloat},
    sampled_representation_joint::Array{<:AbstractFloat},
    sampled_representation_conditionals::Array{<:AbstractFloat},
    joint_exclusion_windows::Array{<:AbstractFloat},
    sampled_joint_exclusion_windows::Array{<:AbstractFloat};
    k::Integer = 4,
    metric::Metric = Euclidean(),
)

    time = joint_exclusion_windows[1, 2, end] - joint_exclusion_windows[1, 1, 1]

    tree_joint = NearestNeighbors.KDTree(representation_joint, metric, reorder = false)

    tree_sampled_joint = NearestNeighbors.KDTree(sampled_representation_joint, metric, reorder = false)

    tree_conditionals = NearestNeighbors.KDTree(representation_conditionals, metric, reorder = false)

    tree_samplel_zonditionals = NearestNeighbors.KDTree(sampled_representation_conditionals, metric, reorder = false)

    l_y = size(representation_joint, 1) - size(representation_conditionals, 1)
    l_x = size(representation_conditionals, 1)

    TE = 0
    for i = 1:size(representation_joint, 2)
        indices_joint, radii_joint = NearestNeighbors.knn(
            tree_joint,
            representation_joint[:, i],
            joint_exclusion_windows[:, :, i],
            joint_exclusion_windows,
            k,
        )

        indices_sampled_joint, radii_sampled_joint = NearestNeighbors.knn(
            tree_sampled_joint,
            representation_joint[:, i],
            joint_exclusion_windows[:, :, i],
            sampled_joint_exclusion_windows,
            k,
        )

        radius_joint = max(maximum(radii_joint), maximum(radii_sampled_joint)) + 1e-6

        indices_conditionals, radii_conditionals = NearestNeighbors.knn(
            tree_conditionals,
            representation_conditionals[:, i],
            joint_exclusion_windows[:, :, i],
            joint_exclusion_windows,
            k,
        )

        indices_samplel_zonditionals, radii_samplel_zonditionals = NearestNeighbors.knn(
            tree_samplel_zonditionals,
            representation_conditionals[:, i],
            joint_exclusion_windows[:, :, i],
            sampled_joint_exclusion_windows,
            k,
        )

        radius_conditionals = max(maximum(radii_conditionals), maximum(radii_samplel_zonditionals)) + 1e-6

        indices_joint = NearestNeighbors.inrange(
            tree_joint,
            representation_joint[:, i],
            joint_exclusion_windows[:, :, i],
            joint_exclusion_windows,
            radius_joint,
        )

        indices_sampled_joint = NearestNeighbors.inrange(
            tree_sampled_joint,
            representation_joint[:, i],
            joint_exclusion_windows[:, :, i],
            sampled_joint_exclusion_windows,
            radius_joint,
        )

        indices_conditionals = NearestNeighbors.inrange(
            tree_conditionals,
            representation_conditionals[:, i],
            joint_exclusion_windows[:, :, i],
            joint_exclusion_windows,
            radius_conditionals,
        )

        indices_samplel_zonditionals = NearestNeighbors.inrange(
            tree_samplel_zonditionals,
            representation_conditionals[:, i],
            joint_exclusion_windows[:, :, i],
            sampled_joint_exclusion_windows,
            radius_conditionals,
        )

        radius_joint = maximum(colwise(metric, representation_joint[:, i], representation_joint[:, indices_joint]))
        radius_sampled_joint =
            maximum(colwise(metric, representation_joint[:, i], sampled_representation_joint[:, indices_sampled_joint]))
        radius_conditionals = maximum(colwise(
            metric,
            representation_conditionals[:, i],
            representation_conditionals[:, indices_conditionals],
        ))
        radius_samplel_zonditionals = maximum(colwise(
            metric,
            representation_conditionals[:, i],
            sampled_representation_conditionals[:, indices_samplel_zonditionals],
        ))

        TE += (
            -(l_x + l_y) * log(2 * radius_joint) +
            (l_x + l_y) * log(2 * radius_sampled_joint) +
            (l_x) * log(2 * radius_conditionals) - (l_x) * log(2 * radius_samplel_zonditionals) +
            digamma(size(indices_joint)[1]) - digamma(size(indices_sampled_joint)[1]) -
            digamma(size(indices_conditionals)[1]) + digamma(size(indices_samplel_zonditionals)[1])
        )

    end

    return (TE / (time))

end



end
