include("../common.jl")
include("../MaskedDiffusion_SP.jl")
include("../methods.jl")
include("../helpers.jl")
include("../ODE.jl")

using DataFrames, CSV
using DifferentialEquations

"""
Solve the ODE for m(t) for each alpha in a given file and save results.

Parameters:
- infile: path to input file with columns α, λ, q, δq, ϕ
- outfile: path to output file (if nothing, auto-generated from infile)
- alphas: range or vector of alpha values to process (if nothing, use all)
- t_0: initial time for backward ODE
- m_t0: initial value m(t_0)
- t_range: range of t values to evaluate and save
"""
function solve_ode_for_alphas(;
    infile,
    outfile=nothing,
    αs=nothing,
    t_0=0.9,
    m_t0=1.0,
    t_range=0.0:0.01:1.0,
    algorithm=:greedy)

    if outfile === nothing
        outfile = replace(infile, ".txt" => "_α_$(αs)_m_t0_$(m_t0)_t0_$(t_0)_$(algorithm).txt")
    end

    df = file_to_df(infile)
    if αs !== nothing
        df = filter(row -> row.α in αs, df)
    end

    output_df = DataFrame()

    # Process each alpha
    println("Processing $(nrow(df)) α values...")
    for (idx, row) in enumerate(eachrow(df))
        α = row.α
        λ = row.λ
        q = row.q
        δq = row.δq

        println("[$idx/$(nrow(df))] α=$(α), q=$(q), δq=$(δq)")

        try
            sol, m_final = solve_m_backward_complete(
                t_0=t_0,
                m_t0=m_t0,
                q=q,
                δq=δq,
                t_final=0.0,
                algorithm=algorithm,
            )

            # Save results for each t in t_range
            for t in t_range
                if t <= t_0
                    m = sol(t)[1]
                    result_row = (; α=α, λ=λ, q=q, δq=δq, t_0, m_t0, t, m)
                    save_row!(result_row, output_df, outfile)
                end
            end

            println("  → Successfully computed m(t) from t=$(t_0) to t=0")
        catch e
            if e isa InterruptException
                rethrow()
            end
            println("  → ERROR: Failed to solve ODE for α=$(α)")
            println("     Error: ", e)
        end
    end

    println("\nResults saved to: $(outfile)")
    return output_df
end

"""
Solve the backward ODE for each alpha and each initial time t_0,
with m_t0 always fixed to 1 - t_0, and save only the final value m_f = m(0).

Parameters:
- infile: path to input file with columns α, λ, q, δq, ϕ
- outfile: path to output file (if nothing, auto-generated from infile)
- αs: range or vector of alpha values to process (if nothing, use all rows)
- t_0_values: iterable of initial times to sweep
- algorithm: :greedy or :fair (passed to solve_m_backward)
- save_trajectory: when true, save one row per sampled time between t_0 and t_final
- trajectory_step: spacing used to sample the saved trajectory when save_trajectory is true

For each (α, t_0) pair, one row is saved with:
α, λ, q, δq, t_0, m_t0, m_f
"""

function solve_mf_dataset(;
    infile,
    outfile=nothing,
    αs=nothing,
    t_0_values=0.0:0.1:1.0,
    algorithm=:greedy,
    save_trajectory::Bool=false,
    trajectory_step::Real=0.01,
    trajectory_times=nothing)

    # Load data first so we can include the actual alpha range in filenames
    df = file_to_df(infile)
    if αs !== nothing
        df = filter(row -> row.α in αs, df)
    end

    # Helper to format numbers for filenames
    fmt(x) = replace(string(round(x, digits=4)), "." => "p")

    if outfile === nothing
        # alpha tag from actual data
        α_min = minimum(df.α)
        α_max = maximum(df.α)
        alpha_tag = "α_$(fmt(α_min))_$(fmt(α_max))"

        if save_trajectory
            step_tag = replace(string(trajectory_step), "." => "p")
            # region tag from t_0_values
            t0_min = minimum(t_0_values)
            t0_max = maximum(t_0_values)
            region_tag = "t0_$(fmt(t0_min))_to_$(fmt(t0_max))"
            outfile = replace(infile, ".txt" => "_m_f_t0_$(algorithm)_trajectory_dt_$(step_tag)_$(region_tag)_$(alpha_tag).txt")
        else
            outfile = replace(infile, ".txt" => "_m_f_t0_$(algorithm)_$(alpha_tag).txt")
        end
    end

    output_df = DataFrame()

    println("Processing $(nrow(df)) α rows and $(length(t_0_values)) t_0 values...")
    for (idx, row) in enumerate(eachrow(df))
        α = row.α
        λ = row.λ
        q = row.q
        δq = row.δq

        println("[$idx/$(nrow(df))] α=$(α), q=$(q), δq=$(δq)")

        for t_0 in t_0_values
            m_t0 = 1 - t_0

            try
                sol_result = solve_m_backward_complete(
                    t_0=t_0,
                    m_t0=m_t0,
                    α=α,
                    q=q,
                    δq=δq,
                    t_final=0.0,
                    algorithm=algorithm,
                    save_trajectory=save_trajectory,
                    trajectory_step=trajectory_step,
                    trajectory_times=trajectory_times,
                )

                if save_trajectory
                    _, m_f, v_f, trajectory = sol_result
                    for (t, m, v) in zip(trajectory.t, trajectory.m, trajectory.v)
                        result_row = (; α=Float64(α), λ=Float64(λ), q=Float64(q), δq=Float64(δq), t_0=Float64(t_0), m_t0=Float64(m_t0), m_f=Float64(m_f), t=Float64(t), m=Float64(m), v=Float64(v))
                        save_row!(result_row, output_df, outfile)
                    end
                else
                    _, m_f, v_f = sol_result
                    result_row = (; α=Float64(α), λ=Float64(λ), q=Float64(q), δq=Float64(δq), t_0=Float64(t_0), m_t0=Float64(m_t0), m_f=Float64(m_f), v_f=Float64(v_f))
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

    println("\nFinal m_f results saved to: $(outfile)")
    return output_df
end
