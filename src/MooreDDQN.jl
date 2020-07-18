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
include("policies.jl")
include("utils.jl")
include("learn.jl")

export learn!, build_DeepQPolicy, action
export get_QValues, get_target, get_params, update_target

end # module
