mutable struct Double_DeepQPolicy <: QPolicy
    primaryNetwork      # Neural Network for Deep Q function
    targetNetwork       # Holds target values - call update_target to copy into this
end
Double_DeepQPolicy(primary) = Double_DeepQPolicy(primary, deepcopy(primary))

get_QValues(policy::Double_DeepQPolicy, inputs; primary=true) = primary ? policy.primaryNetwork(inputs) : policy.targetNetwork(inputs)

function get_target(policy::Double_DeepQPolicy, γ, r, done, s′)
    a_batch = argmax(policy.primaryNetwork(s′), dims=1)
    target = dropdims(r .+ γ.*(1.0 .- done).*policy.targetNetwork(s′)[a_batch], dims=1)
end

get_params(policy::Double_DeepQPolicy) = Flux.params(policy.primaryNetwork)

update_target(policy::Double_DeepQPolicy) = Flux.loadparams!(policy.targetNetwork, Flux.params(policy.primaryNetwork))

action(policy::MooreDDQN.Double_DeepQPolicy, r, s::S, A) where {S} = argmax(policy.primaryNetwork(s()))[1]
