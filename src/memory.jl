struct Experience{T<:Real, V<:AbstractArray{T}, A, F}
    s::V
    a::A
    r::F
    s′::V
    done::Bool
end

abstract type ReplayMemoryBuffer end

mutable struct PriorityReplayMemoryBuffer{T, S<:Sequence} <: ReplayMemoryBuffer
    capacity::Int
    experience::CircularBuffer{Experience}
    priorities::CircularBuffer{T}
    currentMax::T
    batch_size::Int
    α::Float32
    β::S
    ϵ::Float32
end

# Constructor for Empty Memory
function PriorityReplayMemoryBuffer(n::Int, batch_size::Int; β::D, α=5f-1, ϵ=1f-3) where {D<:Sequence}
    PriorityReplayMemoryBuffer(n, CircularBuffer{Experience}(n), CircularBuffer{Float32}(n), ϵ, batch_size, α, β, ϵ)
end

# Utility functions
Base.length(mem::PriorityReplayMemoryBuffer) = length(mem.experience)
Base.size(mem::PriorityReplayMemoryBuffer) = length(mem)

# Memory Control
function addexp!(mem::PriorityReplayMemoryBuffer, Exp::Experience)
    push!(mem.experience, Exp)
    push!(mem.priorities, mem.currentMax)
end

function addexp!(mem::PriorityReplayMemoryBuffer, s::AbstractArray{T}, a::A,
                 r::F, s′::AbstractArray{T}, d::Bool) where {T, A, F}
    addexp!(mem, Experience(s, a, convert(Float32, r), s′, d))
end

# Parameter Control
update_β!(mem::PriorityReplayMemoryBuffer) = step!(mem.β)

# Update priorities for selected indicies - updated while training
function update_priorities!(mem::PriorityReplayMemoryBuffer, ids::Vector{T}, td_errs::V ) where {T<:Int, V <: AbstractArray}
    mem.priorities[ids] = (abs.(td_errs) .+ mem.ϵ).^mem.α
    mem.currentMax = max(mem.currentMax, maximum(mem.priorities[ids]))
end

# Sample from the buffer
function StatsBase.sample(mem::PriorityReplayMemoryBuffer)
    ids = sample(1:length(mem), Weights(mem.priorities) , mem.batch_size, replace=false)
    s = hcat((mem.experience[i].s for i in ids)...)
    r = hcat((mem.experience[i].r for i in ids)...)
    s′ = hcat((mem.experience[i].s′ for i in ids)...)
    d = hcat((mem.experience[i].done for i in ids)...)
    weights = mem.priorities[ids] ./ sum(mem.priorities)
    weights = (length(mem)*weights).^(-value(mem.β))
    weights = weights ./ maximum(weights)

    # Actions need to be converted to Cartesian indices so that they address
    # into the correct place
    a = [CartesianIndex(0,0) for i in ids]
    for (i, idx) in enumerate(ids)
        @inbounds a[i] = CartesianIndex(mem.experience[idx].a, i)
    end
    return (s, a, r, s′, d, ids, weights)
end
