include("common.jl")
include("MaskedDiffusion_SP.jl")
include("methods.jl")
include("helpers.jl")

using DifferentialEquations
using Optimization
using OptimizationOptimJL
using Optim


function trajectory_time_points(t_0, t_final; trajectory_step=0.01)
    step = abs(trajectory_step)
    if step == 0
        error("trajectory_step must be non-zero")
    end

    if isapprox(t_0, t_final)
        return [t_final]
    end

    if t_0 > t_final
        n_steps = Int(floor((t_0 - t_final) / step))
        times = [t_0 - i * step for i in 0:n_steps]
    else
        n_steps = Int(floor((t_final - t_0) / step))
        times = [t_0 + i * step for i in 0:n_steps]
    end

    if isempty(times) || !isapprox(times[end], t_final; atol=step * 1e-8, rtol=0)
        push!(times, t_final)
    end

    return times
end


function solve_m_backward(; t_0, m_t0, q, δq, t_final=0.0, algorithm=:greedy, save_trajectory=false, trajectory_step=0.01, trajectory_times=nothing)
    function dm_dt!(dm, m, p, t)
        # Avoid division by zero at t=0 or t=1
        if t ≈ 0.0 || t ≈ 1.0
            dm[1] = 0.0
            return
        end

        if algorithm == :greedy
            integral = ∫D(x -> begin
                μ_x = Eq.iμ_newton(q, δq, x)
                arg = μ_x * m[1] / √(q * ((1 - t) - m[1]^2))
                return Φ(arg)
            end)
        elseif algorithm == :fair
            integral = ∫D(x -> begin
                μ_x = Eq.iμ_newton(q, δq, x)
                return ∫D(y -> Eq.σ(y * √(q * ((1 - t) - m[1]^2)) + μ_x * m[1]))
            end)
        else
            error("algorithm must be :greedy or :fair")
        end

        dm[1] = 1.0 - 2.0 * integral
    end

    m0 = [m_t0]
    tspan = (t_0, t_final)

    problem = ODEProblem(dm_dt!, m0, tspan)

    solution = solve(problem, Tsit5(), reltol=1e-4, abstol=1e-6)

    m_final = solution(t_final)[1]

    if save_trajectory
        times = trajectory_times === nothing ? trajectory_time_points(t_0, t_final; trajectory_step=trajectory_step) : trajectory_times
        trajectory = (; t=times, m=[solution(t)[1] for t in times])
        return solution, m_final, trajectory
    end

    return solution, m_final
end
function solve_m_backward_complete(;
    t_0,
    m_t0,
    α,
    q,
    δq,
    v_t0=nothing,
    algorithm=:greedy,
    β=1.0,
    t_final=0.0,
    save_trajectory=false,
    trajectory_step=0.01,
    trajectory_times=nothing,
)
    # New fully-Hebbian non-residual closure.
    #
    # μ(x)      = Eq.iμ_newton(q, δq, x)
    # μbar      = E_x[μ(x)]
    # γ2        = E_x[μ(x)^2]
    #
    # Forward equations:
    #
    #   dm/dt = 2 E_{x,z} σ(β[μ(x)m + sqrt(v) z]) - 1
    #
    #   dv/dt = α γ2
    #           - 4 β μbar v E_{x,z} σ'(β[μ(x)m + sqrt(v) z])
    #
    # Greedy β -> ∞:
    #
    #   dm/dt = 2 E_x Φ( μ(x)m / sqrt(v) ) - 1
    #
    #   dv/dt = α γ2
    #           - 4 μbar v E_x[ φ(μ(x)m/sqrt(v)) / sqrt(v) ]
    #
    # Equivalently:
    #
    #   dv/dt = α γ2
    #           - 4 μbar sqrt(v) E_x φ(μ(x)m/sqrt(v)).
    #
    # Since tspan = (t_0, t_final), if t_final < t_0 this solves the
    # same vector field backwards in time.

    if !(algorithm in (:greedy, :fair))
        error("algorithm must be :greedy or :fair")
    end

    μ(x) = Eq.iμ_newton(q, δq, x)

    φ(u) = exp(-0.5 * u^2) / sqrt(2.0 * π)
    σprime(u) = Eq.σ(u) * (1.0 - Eq.σ(u))

    μbar = ∫D(x -> μ(x))
    γ2 = ∫D(x -> μ(x)^2)

    # If no terminal variance is provided, use the no-susceptibility estimate
    # v(t) ≈ α γ2 t as a neutral default.
    v_init = v_t0 === nothing ? α * γ2 * max(t_0, 0.0) : v_t0

    function dm_dv_dt!(du, u, p, t)
        m = u[1]
        v = max(u[2], 0.0)

        if algorithm == :fair
            sqrt_v = sqrt(v)

            mean_response = ∫D(x -> begin
                μx = μ(x)
                ∫D(z -> Eq.σ(β * (μx * m + sqrt_v * z)))
            end)

            slope_integral = ∫D(x -> begin
                μx = μ(x)
                ∫D(z -> σprime(β * (μx * m + sqrt_v * z)))
            end)

            du[1] = 2.0 * mean_response - 1.0
            du[2] = α * γ2 - 4.0 * β * μbar * v * slope_integral

        else
            if v <= 0.0
                # Deterministic greedy limit.
                du[1] = ∫D(x -> sign(μ(x) * m))
                du[2] = α * γ2
            else
                sqrt_v = sqrt(v)

                mean_response = ∫D(x -> begin
                    a = μ(x) * m / sqrt_v
                    Φ(a)
                end)

                susceptibility_integral = ∫D(x -> begin
                    a = μ(x) * m / sqrt_v
                    φ(a) / sqrt_v
                end)

                du[1] = 2.0 * mean_response - 1.0
                du[2] = α * γ2 - 4.0 * μbar * v * susceptibility_integral
            end
        end

        return nothing
    end

    u0 = [m_t0, v_init]
    tspan = (t_0, t_final)

    problem = ODEProblem(dm_dv_dt!, u0, tspan)
    solution = solve(problem, Tsit5(), reltol=1e-4, abstol=1e-6)

    final_state = solution(t_final)
    m_final = final_state[1]
    v_final = final_state[2]

    if save_trajectory
        times = trajectory_times === nothing ?
                trajectory_time_points(t_0, t_final; trajectory_step=trajectory_step) :
                trajectory_times

        trajectory = (;
            t=times,
            m=[solution(t)[1] for t in times],
            v=[solution(t)[2] for t in times],
        )

        return solution, m_final, v_final, trajectory
    end

    return solution, m_final, v_final
end

function solve_mf_hebbian_backward(; t_0, m_t0, α, t_final=0.0, algorithm=:greedy, save_trajectory=false, trajectory_step=0.01, trajectory_times=nothing)
    function dm_dt!(dm, m, p, t)
        if algorithm == :greedy
            ε = Φ(m[1] / sqrt(α))
        elseif algorithm == :fair
            ε = ∫D(z -> Eq.σ(m[1] + sqrt(α * (1 - t)) * z))
        else
            error("algorithm must be :greedy or :fair")
        end

        dm[1] = 1.0 - 2.0 * ε
    end

    m0 = [m_t0]
    tspan = (t_0, t_final)

    problem = ODEProblem(dm_dt!, m0, tspan)
    solution = solve(problem, Tsit5(), reltol=1e-5, abstol=1e-7)

    m_final = solution(t_final)[1]

    if save_trajectory
        times = trajectory_times === nothing ? trajectory_time_points(t_0, t_final; trajectory_step=trajectory_step) : trajectory_times
        trajectory = (; t=times, m=[solution(t)[1] for t in times])
        return solution, m_final, trajectory
    end

    return solution, m_final
end
function solve_mf_hebbian_backward_complete(; t_0, m_t0, α, v_t0=α * max(1.0 - t_0, 0.0), algorithm=:fair, t_final=0.0, save_trajectory=false, trajectory_step=0.01, trajectory_times=nothing)
    # Map algorithms to temperature behavior: greedy -> zero-temperature, fair -> finite (beta=1)
    zero_temperature = algorithm == :greedy

    function dm_dv_dt!(du, u, p, t)
        m = u[1]
        v = max(u[2], 0.0)
        sqrt_v = sqrt(v)

        if zero_temperature
            if sqrt_v == 0.0
                # Limit of -erf(m / sqrt(2v)) and of the susceptibility correction.
                du[1] = -sign(m)
                du[2] = -α
            else
                a = m / sqrt_v
                du[1] = -(2.0 * Φ(a) - 1.0)
                du[2] = -α - 4.0 * sqrt_v / sqrt(2.0 * π) * exp(-0.5 * a^2)
            end
        else
            # finite temperature behavior; use beta=1 for the "fair" algorithm
            b = 1.0
            mean_response = ∫D(z -> tanh(0.5 * b * (m + sqrt_v * z)))
            susceptibility = ∫D(z -> sech(0.5 * b * (m + sqrt_v * z))^2)

            du[1] = -mean_response
            du[2] = -α - b * v * susceptibility
        end
    end

    u0 = [m_t0, v_t0]
    tspan = (t_0, t_final)

    problem = ODEProblem(dm_dv_dt!, u0, tspan)
    solution = solve(problem, Tsit5(), reltol=1e-5, abstol=1e-7)

    final_state = solution(t_final)
    m_final = final_state[1]
    v_final = final_state[2]

    if save_trajectory
        times = trajectory_times === nothing ? trajectory_time_points(t_0, t_final; trajectory_step=trajectory_step) : trajectory_times
        trajectory = (; t=times, m=[solution(t)[1] for t in times], v=[solution(t)[2] for t in times])
        return solution, m_final, v_final, trajectory
    end

    return solution, m_final, v_final
end
