using QuantumInfo
using Test

import LinearAlgebra
import QuantumInfo.eye

@testset "partial trace" begin
    for i=1:100
        da = round(Int,round(rand()*10))+1
        db = round(Int,round(rand()*10))+1
        ra = partialtrace(projector(randn(da*3)+1im*randn(da*3)),[da,3],2)
        rb = partialtrace(projector(randn(db*3)+1im*randn(db*3)),[db,3],2)
        @test isapprox(partialtrace(kron(ra,rb),[da,db],1),rb,atol=1e-15)
        @test isapprox(partialtrace(kron(ra,rb),[da,db],2),ra,atol=1e-15)
    end
end

@testset "representation conversion" begin
    id_choi = 1/2 .* [1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1];
    deph_choi = 1/2 .* [1 0 0 0.99;0 0 0 0;0 0 0 0;0.99 0 0 1];
    deph_liou = [1 0 0 0;0 0.99 0 0;0 0 0.99 0;0 0 0 1];

    # Kraus operators for a dephasing channel
    ρ = 0.01;
    M_0 = sqrt(1-ρ)*eye(2);
    M_1 = sqrt(ρ)*Matrix{ComplexF64}([1.0 0.0; 0.0 0.0]);
    M_2 = sqrt(ρ)*Matrix{ComplexF64}([0.0 0.0;0.0 1.0]);

    @test liou(eye(2),eye(2)) == eye(4)
    @test liou(eye(2)) == liou(eye(2),eye(2))
    @test mat(vec(eye(2))) == eye(2)
    @test mat([1,2,3,4]) == [1 3;2 4]
    @test vec([1 2; 3 4]) == [1,3,2,4]

    @test choi_liou_involution(choi_liou_involution(eye(4)))==eye(4)

    @test isapprox(choi2liou(id_choi),eye(4))
    @test isapprox(liou2choi(eye(4)),id_choi)
    @test isapprox(choi2kraus(id_choi),Matrix{ComplexF64}[zeros(2,2), zeros(2,2), zeros(2,2), eye(2)],atol=1e-7)
    @test isapprox(kraus2liou(Matrix{ComplexF64}[eye(2)]),eye(4))
    @test isapprox(liou2kraus(eye(4)),Matrix{ComplexF64}[zeros(2,2), zeros(2,2), zeros(2,2), eye(2)],atol=1e-7)
    @test isapprox(kraus2choi(Matrix{ComplexF64}[eye(2)]),id_choi)

    @test isapprox(kraus2choi([M_0, M_1, M_2]),deph_choi)
    @test isapprox(kraus2liou([M_0, M_1, M_2]),deph_liou)

    @test isapprox(liou2pauliliou(eye(4)), eye(4))
    @test isapprox(pauliliou2liou(eye(4)), eye(4))
end

@testset "depolarization" begin
    @test isapprox(depol(2),projector(vec(eye(2))),atol=1e-15)
    @test isapprox(depol(2),depol(2,1.))
    @test isapprox(depol(2,0.),eye(4),atol=1e-15)
end

@testset "density matrix properties" begin
  for i=1:100
    rrho = partialtrace(projector(randn(3^3)+1im*randn(3^3)),[3,9],2)
    @test isapprox(LinearAlgebra.tr(rrho),1.)
    @test QuantumInfo.ishermitian(rrho)
    @test ispossemidef(rrho)

    ru = LinearAlgebra.svd(randn(3,3)+1im*randn(3,3)).U
    @test isapprox(ru*ru',eye(3),atol=1e-13)
    @test isapprox(ru'*ru,eye(3),atol=1e-13)

    rv = LinearAlgebra.svd(randn(3,3)+1im*randn(3,3)).U
    re = (liou(ru)+liou(rv))/2
    @test ispossemidef(liou2choi(re),tol=1e-13)
    @test isapprox(partialtrace(liou2choi(re),[3,3],2),eye(3)/3,atol=1e-15)
  end
end
