using SchattenNorms

import Base.trace#, Base.isposdef

export ket,
       bra,
       ketbra,
       trace,
       projector,
       normalize,
       purify,
       fidelity,
       concurrence,
       avg_fidelity,
       ispossemidef

"""
Returns the `(i+1)`th basis element in a `d` dimensional Euclidean space
as a vector of type `t`.
"""
function basis_vector( t::Type, i::Int, d::Int )
  result = zeros(t, d)
  result[i+1] = 1
  result
end

"""
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
function ket(t::Type, label::String, base::Int)
  basis_vector( t, parseint(label, base), base^length(label) )
end
ket( label::String, base=2 ) = ket(Float64, label, base)
ket( t::Type, label::String ) = ket(t, label, 2)

ket( i::Int, d=2 ) = basis_vector( Float64, i, d )

# the duals of the above
"""
Returns a dual vector (row vector) corresponding to a particular basis
unit vector in some Hilbert space.

See `ket` for details on how the corresponding vector may be specified.
"""
function bra( t::Type, label::String, base::Int )
  return ket( t, label, base )'
end

bra( label::String, base::Int ) = bra( Float64, label, base )
bra( label::String ) = bra( Float64, label, 2 )
bra( t::Type, label::String ) = bra( t, label, 2 )
bra( i::Int, d::Int ) = ket( i, d )'

"""
Returns the outer product between a dual vector (row vector) and a vector,
each corresponding to a particular basis unit vector in some Hilbert space.
This can also be interpreted as a matrix unite element specified by two
unit vectors for a row and a column.

See `ket` for details on how the corresponding vectors may be specified.
"""
ketbra( t::Type, label_left::String, label_right::String, base::Int ) = ket( t, label_left, base ) * bra( t, label_right, base )
ketbra( label_left::String, label_right::String, base::Int ) = ket( label_left, base ) * bra( label_right, base )
ketbra( label_left::String, label_right::String ) = ket( label_left ) * bra( label_right )
ketbra( t::Type, label_left::String, label_right::String ) = ket( t, label_left ) * bra( t, label_right )
ketbra( i::Int, j::Int, d::Int ) = ket( i, d ) * bra( j, d )

"""
Computes the rank-1 projector operator corresponding to a given vector. If more than one
vector is given, the projector into the space spanned by the vectors is computed.
"""
function projector( v::Vector )
    v_ = normalize(v)
    v_*v_'
end

"""
Normalizes a vector with respect to the Euclidean norm (L2 norm), or 
a matrix with respect to the trace norm (Schatten 1 norm).
"""
function normalize( v::Vector )
    return v/norm(v)
end

normalize( m::Matrix ) = m/trnorm(m)

# TODO: normalize!( v::Vector ) 
# TODO: normalize!( m::Matrix ) 

# partial trace
"""
Computes the partial trace of a matrix `m`.
"""
function trace{T}( m::Matrix{T}, ds::Vector, dt::Int )
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
    new_m[ii] = trace(reshape(m[ii,:,:],(ds[dt],ds[dt])))
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
function purify{T}( rho::Matrix{T} )
  d = size(rho,1)
  (vals,vecs) = eig(rho)
  psi = zeros(T,d^2)
  for ii=1:d
    psi += sqrt(vals[ii]) * kron(vecs[:,ii],ket(ii-1,d))
  end
  psi
end

# TODO: superoperator purification? in OpenQuantumSystems?

"""
Computes the concurrence of a bipartite qubit state.
"""
function concurrence(m)
    ms = conj(m)
    σy = [0 -1im; 1im 0];
    mt = kron(σy,σy)*ms*kron(σy,σy)
    R = sqrtm(sqrtm(m)*mt*sqrtm(m))
    res = dot(sort(real(eigvals(R)),rev=true),[1,-1,-1,-1])

    return res
end

"""
Computes the fidelity between two quantum states. By default, the Josza convention
is used. The Uhlmann fideilty may be calculated by using the `kind` keyword argument
with the value `:uhlmann`.
"""
function fidelity(a::Matrix,b::Matrix;kind=:josza)
    fu = strnorm(sqrtm(a)*sqrtm(b))
    if kind==:josza
        return fu^2
    elseif kind==:uhlmann
        return fu
    end
end

function fidelity(a::Vector,b::Vector;kind=:josza)
    return fidelity(projector(a),projector(b),kind=kind)
end

function fidelity(a::Vector,b::Matrix;kind=:josza)
    return fidelity(projector(a),b,kind=kind)
end

function fidelity(a::Matrix,b::Vector;kind=:josza)
    return fidelity(b,a,kind=kind)
end

"""
Computes the average fidelity of the outputs of two quantum operations in the Liouville
representation. One of the operations is assumed to be a unitary quantum operation. The 
fidelity measure used is the one defined by Josza.
""" 
function avgfidelity(liou,liou_uni;kind=:josza)
    d = sqrt(size(liou,2))
    f = (real(trace(liou*liou_uni'))+d)/(d^2+d)
    return f
end

# TODO: entfidelity(liou,liou_uni)
#       avg2entfidelity()
#       ent2avgfidelity()

# fidelity Vector Vector
# fidelity Vector Matrix
# fidelity Matrix Matrix
# superfidelity Matrix Matrix
# subfidelity Matrix Matrix

"""
Tests if a matrix is positive semidefinite within a given tolerance.
"""
function ispossemidef(m,tol=0.0)
    evs = eigvals(Hermitian(m))
    tol = tol==0.0 ? eps(eltype(evs)) : tol
    for ev in evs
        if ev < -tol 
            return false
        end
    end
    return true
end
