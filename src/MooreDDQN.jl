module MooreDDQN

using Flux
using Flux.NNlib
using Juno
using DataStructures: CircularBuffer
using BSON: @save, @load
import BSON
using ProgressMeter
using LearnBase: DiscreteSet
using Reinforce
import Reinforce: action, reset!, finished
import StatsBase

include("memory.jl")
export ReplayMemoryBuffer, PriorityReplayMemoryBuffer

include("policies.jl")
export QPolicy, ϵGreedyPolicy, DeepQPolicy, Double_DeepQPolicy, DuelingDouble_DeepQPolicy, reset!

include("utils.jl")

include("learn.jl")
export learn!

end # module
