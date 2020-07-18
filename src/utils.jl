function build_DeepQPolicy(env, num_actions; double=true, dueling=true)
    if double
        if dueling
            # Create a neural network
            # model = Dense(nfields(env.state), num_actions, σ)
            primaryVNetwork = Chain(Dense(nfields(env.state), 10, σ),
                                   Dense(10, 1))
            primaryANetwork = Chain(Dense(nfields(env.state), 10, σ),
                                    Dense(10, num_actions))
            return DuelingDouble_DeepQPolicy(primaryVNetwork, primaryANetwork)
        else
            # Create a neural network
            # model = Dense(nfields(env.state), num_actions, σ)
            primaryNetwork = Chain(Dense(nfields(env.state), 10, σ),
                                   Dense(10, num_actions))
            return Double_DeepQPolicy(primaryNetwork)
        end
    else
        # Create a neural network
        # model = Dense(nfields(env.state), num_actions, σ)
        primaryNetwork = Chain(Dense(nfields(env.state), 10, σ),
                               Dense(10, num_actions))
        # build the Policy
        return DeepQPolicy(primaryNetwork)
    end
end

# Modified from Flux.jl to accept the td error instead of two inputs
function Flux.huber_loss(y; δ=eltype(y)(1))
    abs_error = abs.(y)
    temp = abs_error .< δ
    x = eltype(y)(0.5)
    hub_loss = sum(((abs_error.^2) .* temp) .* x .+ δ*(abs_error .- x*δ) .* (1 .- temp)) * 1 // length(y)
end
