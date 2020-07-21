# Modified from Flux.jl to accept the td error instead of two inputs
function Flux.huber_loss(y; δ=eltype(y)(1))
    abs_error = abs.(y)
    temp = abs_error .< δ
    x = eltype(y)(0.5)
    hub_loss = sum(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp)) * 1 // length(y)
end
