abstract type Decreaser end
value(d::Decreaser) = d.val

struct ConstantDecreaser{T} <: Decreaser
    val::T
end
step!(CD::ConstantDecreaser) = CD.val

mutable struct ExponentialDecreaser{T} <: Decreaser
    val::T
    multiplier::T
end
step!(ED::ExponentialDecreaser) = ED.val *= ED.multiplier

mutable struct HeavisideDecreaser{T} <: Decreaser
    origVal::T  # Value to start with
    lateVal::T  # Value to switch to
    switchIter::Int
    iter::Int
end
HeavisideDecreaser(origVal, lateVal, switchIter, iter = 1) = HeavisideDecreaser(origVal, lateVal, switchIter, iter)
value(HD::HeavisideDecreaser) = HD.iter < HD.switchIter ? HD.origVal : HD.lateVal
function step!(HD::HeavisideDecreaser)
    HD.iter += 1
    HD.iter < HD.switchIter ? HD.origVal : HD.lateVal
end

mutable struct ArithmeticDescreaser{T} <: Decreaser
    val::T
    addVal::T
end
step!(AD::ArithmeticDescreaser) = AD.val += AD.addVal

mutable struct LinearDescreaser{T} <: Decreaser
    startVal::T
    endVal::T
    startIter::Int
    endIter::Int
    currIter::Int
    step::T
    val::T
end
function LinearDescreaser(startVal::T, endVal::T, startIter::Int, endIter::Int, iter = 1) where {T}
    step = (endVal - startVal)/(endIter - startIter)
    val = iter < endIter ? startVal + step*max(iter - startIter, 0) : endVal
    LinearDescreaser(startVal, endVal, startIter, endIter, 1, step, val)
end
function step!(LD::LinearDescreaser)
    LD.currIter += 1
    LD.val = LD.currIter < LD.endIter ? LD.startVal + LD.step*max(LD.currIter - LD.startIter, 0) : LD.endVal
end