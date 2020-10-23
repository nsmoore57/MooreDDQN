mutable struct ϵGreedyPolicy <: Reinforce.AbstractPolicy
    ϵ::Sequence
    greedy::Reinforce.AbstractPolicy
end

action(policy::ϵGreedyPolicy, r, s, A) = only(rand(1)) < value(policy.ϵ) ? rand(A) : action(policy.greedy, r, s, A)

# Leave the ϵ value alone since we want it to be preserved between episodes
reset!(policy::ϵGreedyPolicy) = reset!(policy.greedy)

update_ϵ!(policy::ϵGreedyPolicy) = step!(policy.ϵ)
