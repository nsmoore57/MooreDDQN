# Abstract QLearning Policy
abstract type QPolicy <: Reinforce.AbstractPolicy end
save_policy(policy::QPolicy, filename="model_checkpoint.bson") = BSON.bson(filename, Dict(:policy => policy))
load_policy(filename="model_checkpoint.bson") = BSON.load(filename)[:policy]
Reinforce.reset!(policy::QPolicy) = nothing

# Individual Policies are in the policies folder for easier organization
include("policies/EpGreedyPolicy.jl")
include("policies/DeepQPolicy.jl")
include("policies/DoubleDeepQPolicy.jl")
include("policies/DuelingDouble_DeepQPolicy.jl")
