mutable struct DuelingDouble_DeepQPolicy <: QPolicy
    primaryBaseNetwork   # Neural Netowrk for running before V and A networks
    primaryVNetwork      # Neural Network for Value function
    primaryANetwork      # Neural Network for Advantage function
    targetBaseNetwork    # Holds target values - call update_target to copy into this
    targetVNetwork       # Holds target values - call update_target to copy into this
    targetANetwork       # Holds target values - call update_target to copy into this
end
DuelingDouble_DeepQPolicy(primaryBase, primaryV, primaryA) = DuelingDouble_DeepQPolicy(primaryBase, primaryV, primaryA,
                                                                                       deepcopy(primaryBase), deepcopy(primaryV), deepcopy(primaryA))

function get_QValues(policy::DuelingDouble_DeepQPolicy, inputs; primary=true)
    if primary
        baseresults = policy.primaryBaseNetwork(inputs)
        Aresults = policy.primaryANetwork(baseresults)
        return policy.primaryVNetwork(baseresults) .+ Aresults .- mean(Aresults, dims=1)
    else
        baseresults = policy.targetBaseNetwork(inputs)
        Aresults = policy.targetANetwork(baseresults)
        return policy.targetVNetwork(baseresults) .+ Aresults .- mean(Aresults, dims=1)
    end
end

function get_target(policy::DuelingDouble_DeepQPolicy, γ, r, done, s′)
    a_batch = argmax(get_QValues(policy, s′), dims=1)
    target = dropdims(r .+ γ.*(1.0 .- done).*get_QValues(policy, s′, primary=false)[a_batch], dims=1)
end

get_params(policy::DuelingDouble_DeepQPolicy) = Flux.params(policy.primaryBaseNetwork, policy.primaryVNetwork, policy.primaryANetwork)

function update_target(policy::DuelingDouble_DeepQPolicy)
    Flux.loadparams!(policy.targetBaseNetwork, Flux.params(policy.primaryBaseNetwork))
    Flux.loadparams!(policy.targetVNetwork,    Flux.params(policy.primaryVNetwork))
    Flux.loadparams!(policy.targetANetwork,    Flux.params(policy.primaryANetwork))
end

# Only need to run the advantage network since everything else is the same regardless of action
function action(policy::MooreDDQN.DuelingDouble_DeepQPolicy, r, s::S, A) where {S}
    baseresults = policy.primaryBaseNetwork(s())
    advantage = policy.primaryANetwork(baseresults)
    argmax(advantage)[1]
end
