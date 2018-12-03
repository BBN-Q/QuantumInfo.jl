module QuantumInfo
using Cliffords
using LinearAlgebra: I, norm, tr, eigvals, eigen, svdvals

# As a stopgap, reintroduce the old `eye`.
eye(m::AbstractMatrix) = Matrix{eltype(m)}(I, size(m))

eye(n::Integer) = Matrix{Float64}(I, (n, n))

include("basics.jl")
include("open-systems.jl")

end
