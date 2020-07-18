# Abstract QLearning Policy
abstract type QPolicy <: Reinforce.AbstractPolicy end
save_policy(policy::QPolicy, filename="model_checkpoint.bson") = BSON.bson(filename, Dict(:policy => policy))
load_policy(filename="model_checkpoint.bson") = BSON.load(filename)[:policy]

# \epsilonGreedyPolicy ----------------------------------------------
mutable struct ϵGreedyPolicy <: Reinforce.AbstractPolicy
    ϵ::AbstractFloat
    greedy::Reinforce.AbstractPolicy
end

function action(policy::ϵGreedyPolicy, r, s, A)
    only(rand(1)) < policy.ϵ ? rand(A) : action(policy.greedy, r, s, A)
end

function reset!(policy::ϵGreedyPolicy)
    # Leave the ϵ value alone since we want it to be preserved between episodes
    reset!(policy.greedy)
end


# DeepQPolicy ------------------------------------------------------
mutable struct DeepQPolicy <: QPolicy
    primaryNetwork      # Neural Network for Deep Q function
end

get_QValues(policy::DeepQPolicy, inputs) = policy.primaryNetwork(inputs)
get_target(policy::DeepQPolicy, γ, r, done, s′) = dropdims(r .+ γ.*(1.0 .- done).*maximum(policy.primaryNetwork(s′); dims=1); dims=1)
get_params(policy::DeepQPolicy) = Flux.params(policy.primaryNetwork)


# Double_DeepQPolicy ------------------------------------------------------
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


# DuelingDouble_DeepQPolicy ------------------------------------------------------
mutable struct DuelingDouble_DeepQPolicy <: QPolicy
    primaryVNetwork      # Neural Network for Value function
    primaryANetwork      # Neural Network for Advantage function
    targetVNetwork       # Holds target values - call update_target to copy into this
    targetANetwork       # Holds target values - call update_target to copy into this
end
DuelingDouble_DeepQPolicy(primaryV, primaryA) = DuelingDouble_DeepQPolicy(primaryV, primaryA, deepcopy(primaryV), deepcopy(primaryA))

function get_QValues(policy::DuelingDouble_DeepQPolicy, inputs; primary=true)
    if primary
        return policy.primaryVNetwork(inputs) .+ policy.primaryANetwork(inputs) .- mean(policy.primaryANetwork(inputs), dims=1)
    else
        return policy.targetVNetwork(inputs) .+ policy.targetANetwork(inputs) .- mean(policy.targetANetwork(inputs), dims=1)
    end
end

function get_target(policy::DuelingDouble_DeepQPolicy, γ, r, done, s′)
    a_batch = argmax(get_QValues(policy, s′), dims=1)
    target = dropdims(r .+ γ.*(1.0 .- done).*get_QValues(policy, s′, primary=false)[a_batch], dims=1)
end

get_params(policy::DuelingDouble_DeepQPolicy) = Flux.params(policy.primaryVNetwork, policy.primaryANetwork)

function update_target(policy::DuelingDouble_DeepQPolicy)
    Flux.loadparams!(policy.targetVNetwork, Flux.params(policy.primaryVNetwork))
    Flux.loadparams!(policy.targetANetwork, Flux.params(policy.primaryANetwork))
end
