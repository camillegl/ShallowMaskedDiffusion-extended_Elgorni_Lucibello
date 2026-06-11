include("../common.jl")
include("../methods.jl")
include("../helpers.jl")
include("../ODE.jl")

using DataFrames

"""
Return Hebbian accuracy for a given alpha and t.

For `algorithm=:greedy`, epsilon = Phi(sqrt((1-t)/alpha)).
For `algorithm=:fair`, epsilon = ∫Dz sigma((1-t) + sqrt(alpha*(1-t)) z).
"""
function hebbian_accuracy(alpha, t; algorithm=:greedy)
    alg = Symbol(algorithm)

    if isapprox(t, 1.0; atol=1e-12)
        return 0.5
    end

    if alg == :greedy
        if alpha <= 0
            return 1.0
        end
        return Φ(sqrt((1 - t) / alpha))
    elseif alg == :fair
        sigma(x) = 1 / (1 + exp(-x))
        return ∫D(z -> sigma((1 - t) + sqrt(alpha * (1 - t)) * z))
    else
        error("algorithm must be :greedy or :fair")
    end
end

"""
Create a one-shot Hebbian dataset compatible with plot_oneshot_accuracy.

Output columns:
  α, λ, q, δq, ϕ, t, ε
"""
function generate_hebbian_oneshot_dataset(; outfile,
    αs=0.01:0.01:3.0,
    ts=0.0:0.1:1.0,
    algorithm=:greedy,
    λ=0.0)

    rows = NamedTuple[]
    for α in αs
        for t in ts
            ε = hebbian_accuracy(α, t; algorithm=algorithm)
            push!(rows, (
                α=Float64(α),
                λ=Float64(λ),
                q=NaN,
                δq=NaN,
                ϕ=NaN,
                t=Float64(t),
                ε=Float64(ε),
            ))
        end
    end

    df = DataFrame(rows)
    sort!(df, [:t, :α])
    df_to_file(df, outfile)

    println("Saved one-shot Hebbian dataset to: $(outfile)")
    return df
end

"""
Create an integrated-style Hebbian dataset compatible with plot_integrated_accuracy.

Output columns:
  α, λ, q, δq, t_0, m_t0, m_f

Here m_f is derived from epsilon as: m_f = 1 - 2*t_0*(1-epsilon).
"""
function generate_hebbian_integrated_dataset(; outfile,
    αs=0.01:0.01:3.0,
    t_0_values=0.0:0.1:1.0,
    algorithm=:greedy,
    λ=0.0)

    rows = NamedTuple[]
    for α in αs
        for t_0 in t_0_values
            ε = hebbian_accuracy(α, t_0; algorithm=algorithm)
            m_f = 1 - 2 * t_0 * (1 - ε)
            push!(rows, (
                α=Float64(α),
                λ=Float64(λ),
                q=NaN,
                δq=NaN,
                t_0=Float64(t_0),
                m_t0=Float64(1 - t_0),
                m_f=Float64(m_f),
            ))
        end
    end

    df = DataFrame(rows)
    sort!(df, [:t_0, :α])
    df_to_file(df, outfile)

    println("Saved integrated Hebbian dataset to: $(outfile)")
    return df
end

"""
Solve the integrated Hebbian formula for each alpha and each initial time t_0,
with m_t0 always fixed to 1 - t_0, and save only the final value m_f = m(0).

Uses analytical Hebbian formulas instead of solving the ODE.

Parameters:
- αs: range or vector of alpha values to process
- t_0_values: iterable of initial times to sweep
- algorithm: :greedy or :fair
- outfile: path to output file (if nothing, auto-generated from alphas and algorithm)
- λ: lambda value to embed in output rows (default 0.0)
- outdir: output directory (default "../data")
- save_trajectory: when true, save one row per sampled time between t_0 and t_final
- trajectory_step: spacing used to sample the saved trajectory when save_trajectory is true

For each (α, t_0) pair, one row is saved with:
α, λ, q, δq, t_0, m_t0, m_f
"""
function solve_mf_hebbian_for_alphas_and_t0(;
    αs=0.0:0.05:3.0,
    t_0_values=0.0:0.1:1.0,
    algorithm=:greedy,
    outfile=nothing,
    λ=0.0,
    outdir=joinpath(@__DIR__, "..", "data"),
    save_trajectory::Bool=false,
    trajectory_step::Real=0.01,
    trajectory_times=nothing)

    if outfile === nothing
        fmt(x) = replace(string(round(x, digits=4)), "." => "p")
        α_min = fmt(first(αs))
        α_max = fmt(last(αs))
        t0_min = fmt(first(t_0_values))
        t0_max = fmt(last(t_0_values))
        if save_trajectory
            step_tag = replace(string(trajectory_step), "." => "p")
            outfile = joinpath(outdir, "hebbian_alpha_$(α_min)_$(α_max)_t0_$(t0_min)_$(t0_max)_m_f_t0_$(algorithm)_trajectory_dt_$(step_tag).txt")
        else
            outfile = joinpath(outdir, "hebbian_alpha_$(α_min)_$(α_max)_t0_$(t0_min)_$(t0_max)_m_f_t0_$(algorithm).txt")
        end
    end

    mkpath(dirname(outfile))

    output_df = DataFrame()

    n_α = length(αs)
    n_t0 = length(t_0_values)
    total = n_α * n_t0

    println("Processing $(n_α) α values and $(n_t0) t_0 values (Hebbian ODE)...")
    pair_idx = 0
    for (i, α) in enumerate(αs)
        for (j, t_0) in enumerate(t_0_values)
            pair_idx += 1
            if i == 1 && j == 1
                println("[$pair_idx/$total] α=$(α), t_0=$(t_0)")
            end

            m_t0 = 1 - t_0

            try
                sol_result = solve_mf_hebbian_backward_complete(
                    t_0=t_0,
                    m_t0=m_t0,
                    α=α,
                    t_final=0.0,
                    algorithm=algorithm,
                    save_trajectory=save_trajectory,
                    trajectory_step=trajectory_step,
                    trajectory_times=trajectory_times,
                )

                if save_trajectory
                    _, m_f, v_f, trajectory = sol_result
                    for (t, m, v) in zip(trajectory.t, trajectory.m, trajectory.v)
                        result_row = (; α=Float64(α), λ=Float64(λ), q=NaN, δq=NaN, t_0=Float64(t_0), m_t0=Float64(m_t0), m_f=Float64(m_f), t=Float64(t), m=Float64(m), v=Float64(v))
                        save_row!(result_row, output_df, outfile)
                    end
                else
                    _, m_f, v_f = sol_result
                    result_row = (; α=Float64(α), λ=Float64(λ), q=NaN, δq=NaN, t_0=Float64(t_0), m_t0=Float64(m_t0), m_f=Float64(m_f), v_f=Float64(v_f))
                    save_row!(result_row, output_df, outfile)
                end
            catch e
                if e isa InterruptException
                    rethrow()
                end
                println("  → ERROR: Failed for α=$(α), t_0=$(t_0)")
                println("     Error: ", e)
            end
        end
    end

    println("\nFinal Hebbian m_f results saved to: $(outfile)")
    return output_df
end
