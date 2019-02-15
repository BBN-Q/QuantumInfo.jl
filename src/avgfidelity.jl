abstract type AvgFidelityMetric 

end

convert(t::Type{T},x::AvgFidelityMetric) where {T<:Real} = x.val
promote_rule(t::Type{T},r::Type{R}) where {T<:AvgFidelityMetric,R<:Real} = promote_rule(Float64,R)

"""
AvgFidelity(f;dim=2)
"""
struct AvgFidelity <: AvgFidelityMetric
  dim::Int64
  val::Float64
  AvgFidelity(f;dim=2) = new(dim,f)
end

avgfidelity(x::AvgFidelity) = x.val
convert(t::Type{AvgFidelity},f::AvgFidelity) = AvgFidelity(f.val,dim=f.dim)
convert(t::Type{Float64},f::AvgFidelity) = f.val

"""
DepolRate(p;dim=2)

Depolarizing rate of unitarily invariant twirl of some random channel.
If the unitarily invariant twirl of some channel is `D`, the depolarizing rate `p` is
defined by

`D(ρ) = p ⋅ ρ + (1-p) ⋅ I/d`

where `I` is the identity matrix. Note that this implies `p` may be negative (i.e., it
is not a probability).
"""
struct DepolRate <: AvgFidelityMetric
  dim::Int64
  val::Float64
  DepolRate(p;dim=2) = new(dim,p)
end

avgfidelity(x::DepolRate) = ((x.dim-1)*x.val+1)/x.dim
convert(t::Type{DepolRate},f::AvgFidelity) = DepolRate((f.dim*f.val-1)/(f.dim-1),dim=f.dim)

struct RBDecayRate <: AvgFidelityMetric
  dim::Int64
  val::Float64
  RBDecayRate(p;dim=2) = new(dim,p)
end

avgfidelity(x::RBDecayRate) = 1-x.val
convert(t::Type{RBDecayRate},f::AvgFidelity) = RBDecayRate(1-f.val,dim=f.dim)

struct EntanglementFidelity <: AvgFidelityMetric
  dim::Int64
  val::Float64
  EntanglementFidelity(e; dim=2) = new(dim,e)
end

avgfidelity(x::EntanglementFidelity) = (x.dim * x.val + 1)/(x.dim + 1)
convert(t::Type{EntanglementFidelity},f::AvgFidelity) = EntanglementFidelity((f.val*(f.dim+1)-1)/f.dim,dim=f.dim)

struct χ00 <: AvgFidelityMetric
  dim::Int64
  val::Float64
  χ00(e; dim=2) = new(dim,e)
end

avgfidelity(x::χ00) = (x.dim * x.val + 1)/(x.dim + 1)
convert(t::Type{χ00},f::AvgFidelity) = χ00((f.val*(f.dim+1)-1)/f.dim,dim=f.dim)

# convert{T1,T2}(t::Type{T1},f::T2) = convert(T1,avgfidelity(,dim=f.dim))
