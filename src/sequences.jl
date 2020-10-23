abstract type Sequence end
value(d::Sequence) = d.val
step!(d::Sequence) = d.val

struct ConstantSequence{T} <: Sequence
    val::T
end

mutable struct ExponentialSequence{T} <: Sequence
    val::T
    multiplier::T
end
step!(ED::ExponentialSequence) = ED.val *= ED.multiplier

mutable struct HeavisideSequence{T} <: Sequence
    origVal::T  # Value to start with
    lateVal::T  # Value to switch to
    switchIter::Int
    iter::Int
end
HeavisideSequence(origVal, lateVal, switchIter, iter = 1) = HeavisideSequence(origVal, lateVal, switchIter, iter)
value(HD::HeavisideSequence) = HD.iter < HD.switchIter ? HD.origVal : HD.lateVal
function step!(HD::HeavisideSequence)
    HD.iter += 1
    HD.iter < HD.switchIter ? HD.origVal : HD.lateVal
end

mutable struct ArithmeticSequence{T} <: Sequence
    val::T
    addVal::T
end
step!(AD::ArithmeticSequence) = AD.val += AD.addVal

mutable struct LinearSequence{T} <: Sequence
    startVal::T
    endVal::T
    startIter::Int
    endIter::Int
    currIter::Int
    step::T
    val::T
end
function LinearSequence(startVal::T, endVal::T, endIter::Int; startIter=1, iter = 1) where {T}
    step = (endVal - startVal)/(endIter - startIter)
    val = iter < endIter ? startVal + step*max(iter - startIter, 0) : endVal
    LinearSequence(startVal, endVal, startIter, endIter, 1, step, val)
end

function step!(LD::LinearSequence)
    LD.currIter += 1
    LD.val = LD.currIter < LD.endIter ? LD.startVal + LD.step*max(LD.currIter - LD.startIter, 0) : LD.endVal
end