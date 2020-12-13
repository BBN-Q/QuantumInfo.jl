using Base.Test, QuantumInfo

import Base.isapprox

rtoldefault = Base.rtoldefault

#function isapprox(m1::Matrix,m2::Matrix; rtol::Real=rtoldefault(abs(m1[1,1]),abs(m2[1,1])), atol::Real=0)
#  all(x->isapprox(x,0.0,rtol=rtol,atol=atol),abs(m1-m2))
#end

function isapprox{T1,T2}(v1::Vector{Matrix{T1}},v2::Vector{Matrix{T2}}; rtol::Real=rtoldefault(abs(v1[1][1,1]),abs(v2[1][1,1])), atol::Real=0)
  local v
  if length(v1) != length(v2)
    false
  else
    v = Matrix{T1}[ abs(v1[i]-v2[i]) for i in 1:length(v1) ]
    all(x->isapprox(x,zeros(size(x,1),size(x,2)),rtol=rtol,atol=atol),v)
  end
end

#function ishermitian(m::Matrix; rtol::Real=rtoldefault(abs((m[1,1]+conj(m[1,1]))/2),abs(m[1,1])), atol::Real=0)
#  isapprox((m+m')/2,m,rtol=rtol,atol=atol)
#end

#function ispositive(m::Matrix;rtol::Real=-1,atol::Real=0)
#  if rtol<0
#      ishermitian(m) && all(x->isapprox(x,0,atol=atol),filter(x->x<0,eigvals(Hermitian(m))))
#  else
#      ishermitian(m) && all(x->isapprox(x,0,atol=atol,rtol=rtol),filter(x->x<0,eigvals(Hermitian(m))))
#  end
#end

let

  for i=1:100
    da = round(Int,round(rand()*10))+1
    db = round(Int,round(rand()*10))+1
    ra = partialtrace(projector(randn(da*3)+1im*randn(da*3)),[da,3],2)
    rb = partialtrace(projector(randn(db*3)+1im*randn(db*3)),[db,3],2)
    @test isapprox(partialtrace(kron(ra,rb),[da,db],1),rb,atol=1e-15)
    @test isapprox(partialtrace(kron(ra,rb),[da,db],2),ra,atol=1e-15)
  end

  id_choi = 1/2.*[1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1];

  @test liou(eye(2),eye(2)) == eye(4)
  @test liou(eye(2)) == liou(eye(2),eye(2))
  @test mat(vec(eye(2))) == eye(2)
  @test mat([1,2,3,4]) == [1 3;2 4]
  @test vec([1 2; 3 4]) == [1,3,2,4]

  @test choi_liou_involution(choi_liou_involution(eye(4)))==eye(4)

  @test isapprox(choi2liou(id_choi),eye(4))
  @test isapprox(liou2choi(eye(4)),id_choi)
  @test isapprox(choi2kraus(id_choi),Matrix{Complex128}[zeros(2,2), zeros(2,2), zeros(2,2), eye(2)],atol=1e-7)
  @test isapprox(kraus2liou(Matrix{Complex128}[eye(2)]),eye(4))
  @test isapprox(liou2kraus(eye(4)),Matrix{Complex128}[zeros(2,2), zeros(2,2), zeros(2,2), eye(2)],atol=1e-7)
  @test isapprox(kraus2choi(Matrix{Complex128}[eye(2)]),id_choi)

  @test isapprox(depol(2),projector(vec(eye(2))),atol=1e-15)
  @test isapprox(depol(2),depol(2,1.))
  @test isapprox(depol(2,0.),eye(4),atol=1e-15)

  for i=1:100
    rrho = partialtrace(projector(randn(3^3)+1im*randn(3^3)),[3,9],2)
    @test isapprox(trace(rrho),1.)
    @test ishermitian(rrho)
    @test ispossemidef(rrho)
    
    ru = svd(randn(3,3)+1im*randn(3,3))[1]
    @test isapprox(ru*ru',eye(3),atol=1e-13)
    @test isapprox(ru'*ru,eye(3),atol=1e-13)
    
    rv = svd(randn(3,3)+1im*randn(3,3))[1]
    re = (liou(ru)+liou(rv))/2
    @test ispossemidef(liou2choi(re),tol=1e-13)
    @test isapprox(partialtrace(liou2choi(re),[3,3],2),eye(3)/3,atol=1e-15)
  end

  @test isapprox(liou2pauliliou(eye(4)), eye(4))
  @test isapprox(pauliliou2liou(eye(4)), eye(4))

  @testset "Pauli-Liouville" begin
    # we expect an X90 to transform +X => +X, +Z => -Y, +Y => +Z
    X90 = expm(-1im*π/4*[0 1;1 0])
    @test liou2pauliliou(liou(X90)) ≈ [1  0  0  0;
                                       0  1  0  0;
                                       0  0  0 -1;
                                       0  0  1  0]

    # liou2pauliliou and pauliliou2liou should be inverses
    @test pauliliou2liou(liou2pauliliou(liou(X90))) ≈ liou(X90)
  end

end
