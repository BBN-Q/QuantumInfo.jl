using Cliffords

export mat,
       liou,
       unital,
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
       isunital

mat( v::Vector, r=round(Int,sqrt(length(v))), c=round(Int,sqrt(length(v))) ) = reshape( v, r, c )
liou{T<:AbstractMatrix}( left::T, right::T ) = kron( transpose(right), left )
liou{T<:AbstractMatrix}( m::T ) = kron( conj(m), m )

function choi_liou_involution( r::Matrix )
  d = round(Int, sqrt(size(r,1)) )
  rl = reshape( r, (d, d, d, d) )
  rl = permutedims( rl, [1,3,2,4] )
  reshape( rl, size(r) )
end

function swap_involution( r::Matrix )
  d = round(Int, sqrt(size(r,1)) )
  rl = reshape( r, (d, d, d, d) )
  rl = permutedims( rl, [3,4,1,2] )
  reshape( rl, size(r) )
end

function choi2liou( r::Matrix  )
  sqrt(size(r,1))*choi_liou_involution( r )
end

function liou2choi( r::Matrix )
  choi_liou_involution( r )/sqrt(size(r,1))
end

function choi2kraus{T}( r::Matrix{T}  )
  (vals,vecs) = eig( sqrt(size(r,1))*r )
  #vals = eigvals( sqrt(size(r,1))*r )
  kraus_ops = Matrix{T}[]
  for i in 1:length(vals)
    push!(kraus_ops, sqrt(vals[i])*mat(vecs[:,i]))
  end
  kraus_ops
end

function choi2stinespring{T}( r::Matrix{T}  )
  (vals,vecs) = eig( Hermitian(sqrt(size(r,1))*r) ) # we are assuming Hermiticity-preserving maps
  #vals = eigvals( sqrt(size(r,1))*r )
  A_ops = Matrix{T}[]
  B_ops = Matrix{T}[]
  d = length(vals)
  for i in 1:length(vals)
    push!(A_ops, kron(sqrt(abs(vals[i]))*mat(vecs[:,i]),ket(i-1,d)))
    push!(B_ops, sign(vals[i])*A_ops[end])
  end
  return sum(A_ops),sum(B_ops)
end

function liou2stinespring{T}( r::Matrix{T} )
  return r |> liou2choi |> choi2stinespring
end

function kraus2liou{T}( k::Vector{Matrix{T}} )
  l = zeros(T,map(x->x^2,size(k[1])))
  for i in 1:length(k)
    l = l + liou(k[i],k[i]')
  end
  l
end

function liou2kraus( l::Matrix )
  choi2kraus( liou2choi( l ) )
end

function kraus2choi{T}( k::Vector{Matrix{T}} )
  c = zeros(T,map(x->x^2,size(k[1])))
  for i in 1:length(k)
    c = vec(k[i])*vec(k[i])'
  end
  c/sqrt(size(c,1))
end

# TODO: Add support for sparse matrices
function dissipator( a::Matrix )
  liou(a,a') - 1/2 * liou(a'*a, eye(size(a)...)) - 1/2 * liou(eye(size(a)...),a'*a)
end

function hamiltonian( h::Matrix )
  -1im * ( liou(h,eye(size(h)...)) - liou(eye(size(h)...),h) )
end

_num2quat(n,l) = map(s->parse(Int,s),collect(base(4,n,l)))

function pauliliou2liou( m::Matrix )
  if size(m,1) != size(m,2)
    error("Only square matrices supported")
  elseif size(m,1) != 4^(floor(log2(size(m,1))/2))
    error("Only matrices with dimension 4^n supported.")
  end
  dsq = size(m,1)
  res = zeros(Complex128,size(m))
  l = round(Int,log2(dsq)/2)
  for i=1:dsq
    for j=1:dsq
      res += m[i,j] * vec(complex(Pauli(_num2quat(i-1,l)))) * vec(complex(Pauli(_num2quat(j-1,l))))' / sqrt(dsq)
    end
  end
  res
end

function liou2pauliliou{T}( m::Matrix{T} )
  if size(m,1) != size(m,2)
    error("Only square matrices supported")
  elseif size(m,1) != 4^(floor(log2(size(m,1))/2))
    error("Only matrices with dimension 4^n supported.")
  end
  dsq = size(m,1)
  res = zeros(Complex128,size(m))
  l = round(Int,log2(dsq)/2)
  for i=1:dsq
    for j=1:dsq
      res[i,j] += trace( m * vec(complex(Pauli(_num2quat(j-1,l)))) * vec(complex(Pauli(_num2quat(i-1,l))))' / sqrt(dsq) )
    end
  end
  res
end

"""
Returns a superoperator that replaces the input with a maximally
mixed state with probability p, and leaves it unchanged with probability (1-p).
"""
function depol( d::Int, p=1.0 )
  choi2liou( p * eye(d^2)/d^2 + (1-p) * projector(_max_entangled_state(d)) )
end

"""
Given a superoperator, it extracts the closest superoperator (in Frobenius norm)
that is unital. The result may not be completely positive.
"""
function unitalproj{T}( m::Matrix{T} )
  d2 = size(m,1)
  d  = round(Int,sqrt(d2))
  id = projector(normalize(vec(eye(d))))
  id*m*id + (I-id)*m*(I-id)
end

function iscp(m; tol=0.0)
    evs = eigvals(liou2choi(m))
    tol = tol==0.0 ? eps(abs(one(eltype(m)))) : tol
    all(real(evs) .> -tol) && all(abs(imag(evs)) .< tol)
end

function istp(m; tol=0.0)
    tol = tol==0.0 ? eps(abs(one(eltype(m)))) : tol
    dsq = size(m,1)
    d = round(Int,sqrt(dsq))
    norm(m'*vec(eye(d))-vec(eye(d)),Inf) < tol
end

function ischannel(m; tol=0.0)
    iscp(m,tol=tol) && istp(m,tol=tol)
end

function isunital(m; tol=0.0)
    tol = tol==0.0 ? eps(abs(one(eltype(m)))) : tol
    dsq = size(m,1)
    d = round(Int,sqrt(dsq))
    norm(m*vec(eye(d))-vec(eye(d)),Inf) < tol
end
