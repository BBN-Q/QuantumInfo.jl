export ket,
       bra,
       ketbra,
       partialtrace,
       projector,
       trnormalize,
       trnormalize!,
       purify,
       fidelity,
       concurrence,
       avgfidelity,
       ispossemidef,
       pauli_decomp

trnorm(M::AbstractMatrix) = sum(svdvals(M))

"""
`basis_vector(T,i,d)`

Returns the `(i+1)`th basis element in a `d` dimensional Euclidean space
as a vector of type `t`.
"""
function basis_vector( t::Type{T}, i::Int, d::Int ) where T
  result = zeros(T, d)
  result[i+1] = 1
  result
end

"""
`ket`

Returns a vector corresponding to a particular basis unit vector in some
Hilbert space.

The vector can be specified in a number of different ways:
   + by a string and a base
   + by a integer and a dimension

If the vector is specified by a string and a base, the dimension is
inferred from the length of the string. The base is assumed to be 2
unless explicitly specified.

If the basis element is specified by an integer, the dimension is
assumed to be 2 unless explicitly specified. The integer values
allowed are 0 through the dimension minus 1.

Optionally, the type of the elements of the vector can be specified
in the first argument.
"""
function ket(t::Type, label::AbstractString, base::Int)
  basis_vector( t, parseint(label, base), base^length(label) )
end
ket( label::AbstractString, base=2 ) = ket(Float64, label, base)
ket( t::Type, label::AbstractString ) = ket(t, label, 2)

ket( i::Int, d=2 ) = basis_vector( Float64, i, d )

# the duals of the above
"""
`bra`

Returns a dual vector (row vector) corresponding to a particular basis
unit vector in some Hilbert space.

See `ket` for details on how the corresponding vector may be specified.
"""
function bra( t::Type, label::AbstractString, base::Int )
  return ket( t, label, base )'
end

bra( label::AbstractString, base::Int ) = bra( Float64, label, base )
bra( label::AbstractString ) = bra( Float64, label, 2 )
bra( t::Type, label::AbstractString ) = bra( t, label, 2 )
bra( i::Int, d::Int ) = ket( i, d )'

"""
`ketbra`

Returns the outer product between a dual vector (row vector) and a vector,
each corresponding to a particular basis unit vector in some Hilbert space.
This can also be interpreted as a matrix unite element specified by two
unit vectors for a row and a column.

See `ket` for details on how the corresponding vectors may be specified.
"""
ketbra( t::Type, label_left::AbstractString, label_right::AbstractString, base::Int ) = ket( t, label_left, base ) * bra( t, label_right, base )
ketbra( label_left::AbstractString, label_right::AbstractString, base::Int ) = ket( label_left, base ) * bra( label_right, base )
ketbra( label_left::AbstractString, label_right::AbstractString ) = ket( label_left ) * bra( label_right )
ketbra( t::Type, label_left::AbstractString, label_right::AbstractString ) = ket( t, label_left ) * bra( t, label_right )
ketbra( i::Int, j::Int, d::Int ) = ket( i, d ) * bra( j, d )

"""
`projector(v)`

Computes the rank-1 projector operator corresponding to a given vector. If more than one
vector is given, the projector into the space spanned by the vectors is computed.
"""
function projector( v::AbstractVector )
    v*v'/norm(v,2)^2
end

"""
`trnormalize(m)`

Normalizes a matrix with respect to the trace norm (Schatten 1 norm).
"""
trnormalize( m::AbstractMatrix ) = m/trnorm(m)

"""
`trnormalize!(v)`

Normalizes a matrix with respect to the trace norm (Schatten 1
norm) in place.
"""
function trnormalize!( m::AbstractMatrix{T} ) where T <: ComplexF64
    n = convert(ComplexF64, trnorm(m))
    lmul!(1/n,m)
end

function trnormalize!( m::AbstractMatrix{T} ) where T <: Float64
    n = trnorm(m)
    lmul!(1/n,m)
end

function trnormalize!( m::AbstractMatrix{T} ) where T <: Int64
    n = convert(Float64, trnorm(m))
    m = convert(Array{Float64}, m)
    lmul!(1/n,m)
end

"""
Computes the partial trace of a matrix `m`.
"""
function partialtrace( m::AbstractMatrix{T}, ds::AbstractVector, dt::Int ) where T
  s = size(m)
  l = length(ds);
  if s[1] != s[2]
    error("Partial trace only defined for square matrices.")
  elseif prod(ds) != s[1] || prod(ds) != s[2]
    error("Subsystem decomposition does not match matrix dimensions.")
  end
  # the way we order subsystems is the opposite of the way we order
  # significant digits, so we must changes the index of the subsystem
  ds = reverse(ds)
  dt = l - dt + 1
  m = reshape( m, tuple(ds..., ds...))
  # get a list of the indices not traced over
  indices = [1:l;]
  filter!(x -> x != dt, indices)
  # reshape the matrix so that subsystem to be traced over is
  # the "last" subsystem
  m = permutedims(m, tuple(indices..., map(x->x+l,indices)..., dt, dt+l))
  m = reshape( m, (prod(ds[indices])^2, ds[dt], ds[dt]))
  # now trace over the last subsystem
  new_m = zeros( T, prod(ds[indices])^2 )
  for ii in 1:prod(ds[indices])^2
    new_m[ii] = tr(reshape(m[ii,:,:],(ds[dt],ds[dt])))
  end
  reshape(new_m, (prod(ds[indices]),prod(ds[indices])) )
end

# TODO: specialized, fast versions of partial trace

function _max_entangled_state( d::Int )
  me = zeros(Float64,d^2)
  for ii=0:d-1
    me += kron(ket(ii,d),ket(ii,d))/sqrt(d)
  end
  return me
end

"""
Computes a purification of `rho`
"""
function purify( rho::AbstractMatrix )
  d = size(rho,1)
  (vals,vecs) = eig(rho)
  psi = zeros(eltypeof(rho), d^2)
  for ii=1:d
    psi += sqrt(vals[ii]) * kron(vecs[:,ii],ket(ii-1,d))
  end
  psi
end

"""
Computes the concurrence of a bipartite qubit state.
"""
function concurrence(m)
    ms = conj(m)
    σy = [0 -1im; 1im 0];
    mt = kron(σy,σy)*ms*kron(σy,σy)
    R = sqrt(sqrt(m)*mt*sqrt(m))
    res = dot(sort(real(eigvals(R)),rev=true),[1,-1,-1,-1])

    return res
end

"""
Computes the fidelity between two quantum states. By default, the Josza convention
is used. The Uhlmann fidelity may be calculated by using the `kind` keyword argument
with the value `:uhlmann`.
"""
function fidelity(a::AbstractMatrix,b::AbstractMatrix;kind=:josza)
    fu = trnorm(sqrt(a)*sqrt(b))
    if kind==:josza
        return fu^2
    elseif kind==:uhlmann
        return fu
    end
end

function fidelity(a::AbstractVector,b::AbstractVector;kind=:josza)
    return fidelity(projector(a),projector(b),kind=kind)
end

function fidelity(a::AbstractVector,b::AbstractMatrix;kind=:josza)
    return fidelity(projector(a),b,kind=kind)
end

function fidelity(a::AbstractMatrix,b::AbstractVector;kind=:josza)
    return fidelity(b,a,kind=kind)
end

"""
Computes the average fidelity of the outputs of two quantum operations in the Liouville
representation. One of the operations is assumed to be a unitary quantum operation. The
fidelity measure used is the one defined by Josza.
"""
function avgfidelity(liou,liou_uni;kind=:josza)
    d = sqrt(size(liou,2))
    f = (real(tr(liou*liou_uni'))+d)/(d^2+d)
    return f
end

# TODO: entfidelity(liou,liou_uni)
#       avg2entfidelity()
#       ent2avgfidelity()

# fidelity AbstractVector AbstractVector
# fidelity AbstractVector AbstractMatrix
# fidelity AbstractMatrix AbstractMatrix
# superfidelity AbstractMatrix AbstractMatrix
# subfidelity AbstractMatrix AbstractMatrix

"""
Tests if a matrix is positive semidefinite within a given tolerance.
"""
function ispossemidef(m;tol=0.0)
    evs = eigvals(m)
    tol = tol==0.0 ? 1e2*eps(abs.(one(eltype(m)))) : tol
    all(real(evs) .> -tol) && all(abs.(imag(evs)) .< tol)
end

"""
Decompose a density matrix into Pauli operators with a given cutoff.
"""
function pauli_decomp(ρ::AbstractMatrix, cutoff=1e-3)
    r = Dict{Pauli,Float64}()
    d2 = size(ρ, 1)
    d = round(Int, sqrt(d2))
    for p in allpaulis(d)
        val = real(tr(ρ * p)) / d2
        if val > cutoff
            r[p] = val
        end
    end
    return r
end
