module CoTETE

push!(LOAD_PATH,"NearestNeighbors.jl/src/NearestNeighbors.jl")


using Distances: evaluate, colwise, Metric, Chebyshev, Euclidean
using SpecialFunctions: digamma, gamma

include("preprocessing.jl")

"""
    function do_preprocessing_and_calculate_TE(
        target_events::Array{<:AbstractFloat},
        source_events::Array{<:AbstractFloat},
        d_x::Integer,
        d_y::Integer;
        start_event::Integer = min(d_x, d_y),
        num_target_events::Integer = length(target_events) - start_event,
        num_samples::Integer = num_target_events,
        k::Integer = 5,
        conditioning_events::Array{<:AbstractFloat} = [0.0],
        d_c::Integer = 0,
        metric::Metric = Euclidean(),
        is_surrogate::Bool = false,
        surrogate_upsample_ratio::AbstractFloat = 2.1,
        k_perm::Integer = 5,
        )

    Returns the TE.
"""
function do_preprocessing_and_calculate_TE(
    target_events::Array{<:AbstractFloat},
    source_events::Array{<:AbstractFloat},
    d_x::Integer,
    d_y::Integer;
    start_event::Integer = min(d_x, d_y),
    num_target_events::Integer = length(target_events) - start_event,
    num_samples::Integer = num_target_events,
    k::Integer = 5,
    conditioning_events::Array{<:AbstractFloat} = [0.0],
    d_c::Integer = 0,
    metric::Metric = Euclidean(),
    is_surrogate::Bool = false,
    surrogate_upsample_ratio::AbstractFloat = 2.1,
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
        d_x,
        d_y,
        num_target_events = num_target_events,
        num_samples = num_samples,
        start_event = start_event,
        conditioning_events = conditioning_events,
        d_c = d_c,
        is_surrogate = is_surrogate,
        surrogate_upsample_ratio = surrogate_upsample_ratio,
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

    tree_sampled_conditionals = NearestNeighbors.KDTree(sampled_representation_conditionals, metric, reorder = false)

    d_y = size(representation_joint, 1) - size(representation_conditionals, 1)
    d_x = size(representation_conditionals, 1)

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

        indices_sampled_conditionals, radii_sampled_conditionals = NearestNeighbors.knn(
            tree_sampled_conditionals,
            representation_conditionals[:, i],
            joint_exclusion_windows[:, :, i],
            sampled_joint_exclusion_windows,
            k,
        )

        radius_conditionals = max(maximum(radii_conditionals), maximum(radii_sampled_conditionals)) + 1e-6

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

        indices_sampled_conditionals = NearestNeighbors.inrange(
            tree_sampled_conditionals,
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
        radius_sampled_conditionals = maximum(colwise(
            metric,
            representation_conditionals[:, i],
            sampled_representation_conditionals[:, indices_sampled_conditionals],
        ))

        TE += (
            -(d_x + d_y) * log(2 * radius_joint) +
            (d_x + d_y) * log(2 * radius_sampled_joint) +
            (d_x) * log(2 * radius_conditionals) - (d_x) * log(2 * radius_sampled_conditionals) +
            digamma(size(indices_joint)[1]) - digamma(size(indices_sampled_joint)[1]) -
            digamma(size(indices_conditionals)[1]) + digamma(size(indices_sampled_conditionals)[1])
        )

    end

    return (TE / (time))

end



end
