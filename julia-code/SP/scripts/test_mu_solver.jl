include("../MaskedDiffusion_SP.jl")
include("../scripts/plots.jl")

using BenchmarkTools

df = file_to_df("data/λ001_α01_10_051125_1211.txt")


function iμ_newton(op, x, μ_0)
    return newton(μ -> begin
            q = op.q
            δq = op.δq
            return -μ / δq + x * √q / δq - Eq.∂μ_l(q, μ)
        end, μ_0, NewtonMethod(dx=1e-6, verb=0, atol=1e-8))
end

function test_results(; n=10000, verbose=false, tol=1e-6)
    μ_0 = 4.0
    identical_count = 0
    similar_count = 0

    for i in 1:n
        j = rand(1:nrow(df))
        q = df.q[j]
        δq = df.δq[j]
        op = Eq.OrderParams(q, δq)
        x = randn()
        verbose && println("Benchmarking BLFGS...")
        μ_BLFGS = optimize(μ -> -Eq.fₑ(op.q, op.δq, μ[1], x), [0.], Optim.LBFGS()).minimizer[1]
        verbose && println("Benchmarking iμ_newton...")
        _, μ_newton = iμ_newton(op, x, μ_0)

        diff = abs(μ_newton - μ_BLFGS)

        if diff < eps(Float64)
            identical_count += 1
        elseif diff < tol
            similar_count += 1
        end

        verbose && println("newton μ: $μ_newton, BLFGS μ: $μ_BLFGS, diff: $diff")
    end

    println("\nResults for $n tests:")
    println("Identical (within machine precision): $identical_count ($(round(identical_count/n*100, digits=2))%)")
    println("Similar (within tolerance $tol): $similar_count ($(round(similar_count/n*100, digits=2))%)")
    println("Different: $(n - identical_count - similar_count) ($(round((n - identical_count - similar_count)/n*100, digits=2))%)")

    return (identical=identical_count, similar=similar_count, different=n - identical_count - similar_count)
end


q = df.q[50]
δq = df.δq[50]
op = Eq.OrderParams(q, δq)
_, μ_newton = iμ_newton(op, 10, 5.)

test_results(n=1000)

function test_times(; n=10)
    μ_0 = 4.0

    for i in 1:n
        j = rand(1:nrow(df))
        q = df.q[j]
        δq = df.δq[j]
        op = Eq.OrderParams(q, δq)
        x = randn()
        println("Timing for q=$q, δq=$δq, x=$x")
        print("Benchmarking BLFGS...")
        @btime optimize(μ -> -Eq.fₑ(op.q, op.δq, μ[1], x), [0.], Optim.LBFGS()).minimizer[1]
        print("Benchmarking iμ_newton...")
        @btime iμ_newton(op, x, μ_0)
    end
end

test_times()