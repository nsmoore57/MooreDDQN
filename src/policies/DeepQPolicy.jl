mutable struct DeepQPolicy <: QPolicy
    primaryNetwork      # Neural Network for Deep Q function
end

get_QValues(policy::DeepQPolicy, inputs) = policy.primaryNetwork(inputs)

get_target(policy::DeepQPolicy, γ, r, done, s′) = dropdims(r .+ γ.*(1.0 .- done).*maximum(policy.primaryNetwork(s′); dims=1); dims=1)

get_params(policy::DeepQPolicy) = Flux.params(policy.primaryNetwork)

action(policy::MooreDDQN.DeepQPolicy, r, s::S, A) where {S <: Reinforce.AbstractState} = argmax(policy.primaryNetwork(s()))[1]
