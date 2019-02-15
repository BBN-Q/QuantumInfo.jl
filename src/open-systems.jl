export mat,
       liou,
       unitalproj,
       choi_liou_involution,
       swap_involution,
       choi2liou,
       choi2kraus,
       choi2stinespring,
       kraus2choi,
       kraus2liou,
       kraus2stinespring,
       liou2choi,
       liou2kraus,
       liou2stinespring,
       liou2pauliliou,
       pauliliou2liou,
       dissipator,
       hamiltonian,
       depol,
       istp,
       iscp,
       ischannel,
       isunital,
       isliouvillian,
       nearestu

mat(v::AbstractVector, r=round(Int,sqrt(length(v))), c=round(Int,sqrt(length(v))) ) = reshape( v, r, c )
liou( left::AbstractMatrix, right::AbstractMatrix )  = kron( transpose(right), left )
liou( m::AbstractMatrix ) = kron( conj(m), m )

function choi_liou_involution( r::AbstractMatrix )
  d = round(Int, sqrt(size(r,1)) )
  rl = reshape( r, (d, d, d, d) )
  rl = permutedims( rl, [1,3,2,4] )
  reshape( rl, size(r) )
end

function swap_involution( r::AbstractMatrix )
  d = round(Int, sqrt(size(r,1)) )
  rl = reshape( r, (d, d, d, d) )
  rl = permutedims( rl, [3,4,1,2] )
  reshape( rl, size(r) )
end

function choi2liou( r::AbstractMatrix  )
  sqrt(size(r,1))*choi_liou_involution( r )
end

function liou2choi( r::AbstractMatrix )
  choi_liou_involution( r )/sqrt(size(r,1))
end

function choi2kraus( r::AbstractMatrix  )
    r = eigen( sqrt(size(r,1))*r )
    vals, vecs = (r.values, r.vectors)
    #vals = eigvals( sqrt(size(r,1))*r )
    kraus_ops = Matrix{eltype(r)}[]
    for i in 1:length(vals)
        push!(kraus_ops, sqrt(vals[i])*mat(vecs[:,i]))
    end
    kraus_ops
end

function choi2stinespring( r::AbstractMatrix  )
  r = eigen( Hermitian(sqrt(size(r,1))*r) ) # we are assuming Hermiticity-preserving maps
  vals, vecs = (r.values, r.vectors)
    #vals = eigvals( sqrt(size(r,1))*r )
  A_ops = Matrix{eltype(r)}[]
  B_ops = Matrix{eltype(r)}[]
  d = length(vals)
  for i in 1:length(vals)
    push!(A_ops, kron(sqrt(abs(vals[i]))*mat(vecs[:,i]),ket(i-1,d)))
    push!(B_ops, sign(vals[i])*A_ops[end])
  end
  return sum(A_ops),sum(B_ops)
end

function liou2stinespring( r::AbstractMatrix )
  return r |> liou2choi |> choi2stinespring
end

function kraus2liou( k::AbstractVector )
  l = zeros(eltype(k[1]),map(x->x^2,size(k[1])))
  for i in 1:length(k)
    l = l + liou(k[i],k[i]')
  end
  l
end

function liou2kraus( l::AbstractMatrix )
  choi2kraus( liou2choi( l ) )
end

function kraus2choi( k::AbstractVector )
  c = zeros(eltype(k[1]),map(x->x^2,size(k[1])))
  for i in 1:length(k)
    c = c + vec(k[i])*vec(k[i])'
  end
  c/sqrt(size(c,1))
end

# TODO: Add support for sparse matrices
function dissipator( a::AbstractMatrix )
  liou(a,a') - 1/2 * liou(a'*a, eye(a)) - 1/2 * liou(eye(a),a'*a)
end

function hamiltonian( h::AbstractMatrix )
  -1im * ( liou(h,eye(h)) - liou(eye(h),h) )
end

function pauliliou2liou( m::AbstractMatrix )
  if size(m,1) != size(m,2)
    error("Only square matrices supported")
  elseif size(m,1) != 4^(floor(log2(size(m,1))/2))
    error("Only matrices with dimension 4^n supported.")
  end
  dsq = size(m,1)
  res = zeros(ComplexF64,size(m))
  n = round(Int,log(2,dsq)/2)
  for (i,pi) in enumerate(allpaulis(n))
    for (j,pj) in enumerate(allpaulis(n))
      res += m[i,j] * vec(complex(pi)) * vec(complex(pj))' / sqrt(dsq)
    end
  end
  res
end

function liou2pauliliou( m::AbstractMatrix )
  if size(m,1) != size(m,2)
    error("Only square matrices supported")
  elseif size(m,1) != 4^(floor(log2(size(m,1))/2))
    error("Only matrices with dimension 4^n supported.")
  end
  dsq = size(m,1)
  res = zeros(ComplexF64,size(m))
  n = round(Int,log(2,dsq)/2)
  for (i,pi) in enumerate(allpaulis(n))
    for (j,pj) in enumerate(allpaulis(n))
      res[i,j] += tr( m * vec(complex(pi)) * vec(complex(pj))' / sqrt(dsq) )
    end
  end
  res
end

"""
`depol(d, p=1.0)`

`depol` return a superoperator that replaces the input with a maximally
mixed state with probability p, and leaves it unchanged with probability (1-p).
"""
function depol( d::Int, p=1.0 )
  choi2liou( p * eye(d^2)/d^2 + (1-p) * projector(_max_entangled_state(d)) )
end

"""
`unitalproj(m)`

Given a superoperator `j` in a Liouville representation, `unitalproj`
extracts the closest superoperator (in Frobenius norm) that is
unital. The result may not be completely positive.
"""
function unitalproj( m::AbstractMatrix )
  d2 = size(m,1)
  d  = round(Int,sqrt(d2))
  id = projector(normalize(vec(eye(d))))
  id*m*id + (I-id)*m*(I-id)
end

"""
ishermitian(m; tol)

Checks if a matrix is Hermitian.
"""
function ishermitian(m; tol=0.0)
    ah = (m-m')/2
    tol = tol==0.0 ? 1e2*eps(abs(one(eltype(m)))) : tol
    norm(ah,Inf)<tol
end

"""
iscp(m; tol)

Checks if the liouville representation of a map (in the natural, computational basis) is completely positive.
"""
function iscp(m; tol=0.0)
    ispossemidef(liou2choi(m))
end

"""
istp(m; tol)

Checks if the Liouville representation of a map is trace preserving (TP).
"""
function istp(m; tol=0.0)
    tol = tol==0.0 ? 1e2*eps(abs(one(eltype(m)))) : tol
    dsq = size(m,1)
    d = round(Int,sqrt(dsq))
    norm(m'*vec(eye(d))-vec(eye(d)),Inf) < tol
end

"""
ischannel(m; tol)

Checks if the Liouville representation of a map is completely positive (CP) and trace preserving (TP).
"""
function ischannel(m; tol=0.0)
    iscp(m,tol=tol) && istp(m,tol=tol)
end

"""
isunital(m; tol)

Checks the conditions for unitality of a map in Liouville representation.
"""
function isunital(m; tol=0.0)
    tol = tol==0.0 ? 1e2*eps(abs(one(eltype(m)))) : tol
    dsq = size(m,1)
    d = round(Int,sqrt(dsq))
    norm(m*vec(eye(d))-vec(eye(d)),Inf) < tol
end

"""
isliouvillian(m; tol)

Checks the conditions for a physical Liouvillian matrix (CPTP map generator)
"""
function isliouvillian(m;tol=0.0)
    tol = tol==0.0 ? 1e2*eps(abs(one(eltype(m)))) : tol

    mΓ = choi_liou_involution(m)
    d = round(Int,sqrt(size(m,1)))
    ω = sum([kron(ket(i,d),ket(i,d)) for i in 0:d-1])/sqrt(d)
    Πω = projector(ω)
    Πω⊥ = eye(d^2)-Πω

    return ishermitian(mΓ,tol=tol) && norm(ω'*m,Inf)<tol && ispossemidef(Πω⊥*mΓ*Πω⊥,tol=tol)
end

"""
liouvillian_violation(m; tol)

Computes violation of conditions for a physical Liouvillian matrix (CPTP map generator)
"""
function liouvillian_violation(m;tol=0.0)
    tol = tol==0.0 ? 1e2*eps(abs(one(eltype(m)))) : tol

    mΓ = choi_liou_involution(m)
    d = round(Int,sqrt(size(m,1)))
    ω = sum([kron(ket(i,d),ket(i,d)) for i in 0:d-1])/sqrt(d)
    Πω = projector(ω)
    Πω⊥ = eye(d^2)-Πω

    return (ahermpart(mΓ), ω'*m, Πω⊥*mΓ*Πω⊥)
end

"""
nearestu(l)

Computes the unitary CP map closest to a given CP map in an interferometric sense.
See D. Oi, [Phys. Rev. Lett. 91, 067902 (2003)](http://journals.aps.org/prl/abstract/10.1103/PhysRevLett.91.067902)
"""
function nearestu(l)
    c = liou2choi(l)
    r = eigen(Hermitian(c))
    vals, vecs = (r.values, r.vectors)
    imax = argmax(vals)
    Λ = mat(vecs[:,imax])
    U,Σ,V = svd(Λ)
    W = U*V'
    return kron(conj(W),W)
end
