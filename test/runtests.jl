using QuantumInfo
using Test

using LinearAlgebra: Diagonal
import LinearAlgebra
import QuantumInfo.eye

@time @testset "partial trace" begin
    for i=1:100
        da = round(Int,round(rand()*10))+1
        db = round(Int,round(rand()*10))+1
        ra = partialtrace(projector(randn(da*3)+1im*randn(da*3)),[da,3],2)
        rb = partialtrace(projector(randn(db*3)+1im*randn(db*3)),[db,3],2)
        @test isapprox(partialtrace(kron(ra,rb),[da,db],1),rb,atol=1e-15)
        @test isapprox(partialtrace(kron(ra,rb),[da,db],2),ra,atol=1e-15)
    end
end

alteye(m::AbstractMatrix) = Diagonal(ones(eltype(m), size(m, 1)))
alteye(n::Integer) = Diagonal(ones(n))
alteye(T::DataType, n::Integer) = Diagonal(ones(T, n))

for eyesym in (:eye, :alteye)
    println("identity matrix is $(string(eyesym))")
    @eval begin
        @time @testset "representation conversion" begin
            id_choi = 1/2 .* [1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1];
            deph_choi = 1/2 .* [1 0 0 0.99;0 0 0 0;0 0 0 0;0.99 0 0 1];
            deph_liou = [1 0 0 0;0 0.99 0 0;0 0 0.99 0;0 0 0 1];

            # Kraus operators for a dephasing channel
            ρ = 0.01;
            M_0 = sqrt(1-ρ)*($eyesym)(2);
            M_1 = sqrt(ρ)*Matrix{ComplexF64}([1.0 0.0; 0.0 0.0]);
            M_2 = sqrt(ρ)*Matrix{ComplexF64}([0.0 0.0;0.0 1.0]);

            @test liou(($eyesym)(2),($eyesym)(2)) == ($eyesym)(4)
            @test liou(($eyesym)(2)) == liou(($eyesym)(2),($eyesym)(2))
            @test mat(vec(($eyesym)(2))) == ($eyesym)(2)
            @test mat([1,2,3,4]) == [1 3;2 4]
            @test vec([1 2; 3 4]) == [1,3,2,4]

            @test choi_liou_involution(choi_liou_involution(($eyesym)(4)))==($eyesym)(4)

            @test isapprox(choi2liou(id_choi),($eyesym)(4))
            @test isapprox(liou2choi(($eyesym)(4)),id_choi)
            @test isapprox(choi2kraus(id_choi),Matrix{ComplexF64}[zeros(2,2), zeros(2,2), zeros(2,2), ($eyesym)(2)],atol=1e-7)
            @test isapprox(kraus2liou(Matrix{ComplexF64}[($eyesym)(2)]),($eyesym)(4))
            @test isapprox(liou2kraus(($eyesym)(4)),Matrix{ComplexF64}[zeros(2,2), zeros(2,2), zeros(2,2), ($eyesym)(2)],atol=1e-7)
            @test isapprox(kraus2choi(Matrix{ComplexF64}[($eyesym)(2)]),id_choi)

            @test isapprox(kraus2choi([M_0, M_1, M_2]),deph_choi)
            @test isapprox(kraus2liou([M_0, M_1, M_2]),deph_liou)

            @test isapprox(liou2pauliliou(($eyesym)(4)), ($eyesym)(4))
            @test isapprox(pauliliou2liou(($eyesym)(4)), ($eyesym)(4))
        end
    end
end

@testset "Pauli-Liouville" begin
    # we expect an X90 to transform +X => +X, +Z => -Y, +Y => +Z
    X90 = exp(-1im*π/4*[0 1;1 0])
    @test liou2pauliliou(liou(X90)) ≈ [1 0 0 0;
                                       0 1 0 0;
                                       0 0 0 -1;
                                       0 0 1 0]

    # liou2pauliliou and pauliliou2liou should be inverses
    @test pauliliou2liou(liou2pauliliou(liou(X90))) ≈ liou(X90)
end

@time @testset "depolarization" begin
    @test isapprox(depol(2),projector(vec(eye(2))),atol=1e-15)
    @test isapprox(depol(2),depol(2,1.))
    @test isapprox(depol(2,0.),eye(4),atol=1e-15)
end

@time @testset "density matrix properties" begin
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
