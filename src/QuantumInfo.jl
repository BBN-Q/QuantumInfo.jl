module QuantumInfo
using Cliffords
import LinearAlgebra

# As a stopgap, reintroduce the old `eye`.
eye(m::AbstractMatrix) = Matrix{eltype(m)}(LinearAlgebra.I, size(m))

eye(n::Integer) = Matrix{Float64}(LinearAlgebra.I, (n, n))

include("basics.jl")
include("open-systems.jl")

end
