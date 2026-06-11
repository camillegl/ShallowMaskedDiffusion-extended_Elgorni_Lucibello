include("../common.jl")
include("../MaskedDiffusion_SP.jl")
include("../methods.jl")
include("../helpers.jl")

using Plots, Statistics, Printf
using DataFrames, CSV
using DelimitedFiles
using LaTeXStrings
using FiniteDifferences
using Optimization
using OptimizationOptimJL
using Optim

const _PLOT_NAME_DIGITS = 4

fmt_plot_value(x) = replace(string(round(x, digits=_PLOT_NAME_DIGITS)), "." => "p")

function plot_colors(n, scheme=:viridis)
    n <= 0 && return Any[]
    gradient = cgrad(scheme, n, categorical=true)
    return [gradient[i] for i in 1:n]
end

sorted_subset(df, predicate, sortcol) = sort(filter(predicate, df), sortcol)

function save_plot(p, outdir, filename)
    mkpath(outdir)
    savefig(p, joinpath(outdir, filename))
    return p
end

function plot_q_and_dq_vs_alpha(df; outdir="plots")
    λs = sort(unique(df.λ))
    colors = plot_colors(length(λs))

    vals = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 2, 4, 5, 10]
    # q vs alpha
    p1 = plot(title="q vs α", xlabel=L"\alpha", ylabel=L"q", legend=:right)
    p2 = plot(title="δq vs α", xlabel=L"\alpha", ylabel=L"\delta q", legend=:right)
    p3 = plot(title="q + δq vs α", xlabel=L"\alpha", ylabel=L"q + δq", legend=:right)
    p4 = plot(title="ϕ vs α", xlabel=L"\alpha", ylabel=L"\phi", legend=:right)
    p5 = plot(title="q and ϕ vs α", xlabel=L"\alpha", ylabel="", legend=:topright, xscale=:log10, xticks=(vals, string.(vals)), xlim=[0.1, 10])
    for (i, λ) in enumerate(λs)
        df_λ = filter(row -> row.λ == λ, df)
        plot!(p1, df_λ.α, df_λ.q, label="λ=$λ", color=colors[i], lw=2, alpha=0.5)
        plot!(p2, df_λ.α, df_λ.δq, label="λ=$λ", color=colors[i], lw=2, alpha=0.5)
        plot!(p3, df_λ.α, df_λ.δq .+ df_λ.q, label="λ=$λ", color=colors[i], lw=2, alpha=0.5)
        plot!(p3, df_λ.α, df_λ.q, label="", color=colors[i], lw=2, alpha=0.5)
        plot!(p4, df_λ.α, df_λ.ϕ, label="λ=$λ", color=colors[i], lw=2, alpha=0.5)
        plot!(p5, df_λ.α, df_λ.q, label="q", color=:lightblue, lw=3, alpha=1)
        plot!(p5, df_λ.α, -(df_λ.ϕ .- df_λ.λ .* df_λ.q) ./ df.α, label="Loss", color=:orange, lw=3, alpha=1)
        # plot!(p5, df_λ.α, -df_λ.ϕ, label="Loss", color=:orange, lw=3, alpha=1)
    end


    # display(p1)
    # display(p2)
    # display(p3)
    # display(p4)
    display(p5)
end

function plot_ε_vs_alpha(df; outdir="plots", save=false, fname="varepsilon_vs_alpha.png", λ, approx=false)
    ts = sort(unique(df.t))
    colors = plot_colors(length(ts))

    p = plot(title="ε vs α for different t", xlabel=L"\alpha", ylabel=L"\varepsilon", legend=:topright, dpi=600)

    for (i, tv) in enumerate(ts)
        df_t = filter(row -> row.t == tv, df)
        # ensure rows are sorted by α for a clean line
        sort!(df_t, :α)
        plot!(p, df_t.α, df_t.ε, label="t=$(tv)", color=colors[i], lw=2)
        if approx && i == length(ts)
            μ̂₀ = optimize(μ -> -Eq.fₑ(0., 1 / λ, μ[1], 0.), [0.], Optim.LBFGS()).minimizer[1]
            Aμ = ∫d(t -> Eq.σ((1 - t) * μ̂₀) * (1 - Eq.σ((1 - t) * μ̂₀)) * (1 - t)^2, [0, 1])
            χμ = 1 / (1 + Aμ / λ)
            χq = ∫d(t -> Eq.σ((1 - t) * μ̂₀) * (1 - Eq.σ((1 - t) * μ̂₀)) * (1 - t) * t, [0, 1])
            plot!(df_t.α, Φ.(1 ./ 4 .* 1 ./ (λ .* (1 .+ df_t.α .* (χμ + χq) ./ λ) .* sqrt.(df_t.α .* (μ̂₀ .^ 2 .+ df_t.α ./ μ̂₀ .^ 2)))), lw=3, linestyle=:dash, label="Approximation", color=:tomato)
        end
    end

    if save
        save_plot(p, outdir, fname)
    end
    return p
end


function plot_ε₀_approx(; λ, α=0:0.01:2)

    λs = sort(λ)
    colors = plot_colors(length(λs), :roma)

    plt = plot()
    for (i, l) in enumerate(λs)
        μ̂₀ = optimize(μ -> -Eq.fₑ(0., 1 / l, μ[1], 0.), [0.], Optim.LBFGS()).minimizer[1]
        Aμ = ∫d(t -> Eq.σ((1 - t) * μ̂₀) * (1 - Eq.σ((1 - t) * μ̂₀)) * (1 - t)^2, [0, 1])
        χμ = Aμ / (1 + Aμ / l)
        χq = ∫d(t -> Eq.σ((1 - t) * μ̂₀) * (1 - Eq.σ((1 - t) * μ̂₀)) * (1 - t) * t, [0, 1])
        plot!(α, Φ.(1 ./ 4 .* 1 ./ (l .* (1 .+ α .* (χμ + χq) ./ l) .* sqrt.(α .* (μ̂₀ .^ 2 .+ α ./ μ̂₀ .^ 2)))), lw=3, label="λ=$l", color=colors[i])
    end
    display(plt)
end

function plot_t_star_vs_alpha(df; δ)
    colors = plot_colors(length(δ), :solar)

    plt = plot(xlabel="α", ylabel="t_star")

    for (i, d) in enumerate(δ)
        df_t = filter(r -> 1 - r.ε < d, df)
        dfmax = combine(groupby(df_t, :t), :α => maximum => :α)
        plot!(dfmax.α, dfmax.t, color=colors[i], lw=2, label="δ=$(d)")
    end
    display(plt)
end


function plot_t_star_vs_alpha_m(df; δ)
    colors = plot_colors(length(δ), :solar)

    plt = plot(xlabel="α", ylabel="t_star")

    for (i, d) in enumerate(δ)
        df_t = filter(r -> 1 - r.m < d, df)
        dfmax = combine(groupby(df_t, :t_0), :α => maximum => :α)
        plot!(dfmax.α, dfmax.t_0, color=colors[i], lw=2, label="δ=$(d)")
    end
    display(plt)
end

# Convenience loader for span files matching pattern and plotting t_star using m(t=0)
function plot_t_star_vs_alpha_span(; δ, dir="data", pattern="span_λ001_α001_3_241125_1631_t")
    files = filter(f -> startswith(f, pattern), readdir(dir; join=true))
    dfs = map(file_to_df, files)
    df_all = vcat(dfs...)
    plot_t_star_vs_alpha_m(df_all; δ)
end

function plot_μ_vs_x(df; λ)
    αs = sort(unique(df.α))
    colors = plot_colors(length(αs))

    plt = plot(title="μ(x) at t = 0", xlabel="x", ylabel="μ", legend=:bottomright, ylim=[4.2, 4.22], xlim=[-1, 1])

    for (i, a) in enumerate(unique(df.α))
        df_a = filter(:α => ==(a), df)
        sort!(df_a, :x)

        μ₀ = df_a.μ[findfirst(df_a.x .== 0)]
        Aμ = ∫d(t -> ∫D(y -> Eq.σ(sqrt(t * (1 - t) * a) * y / μ₀ + (1 - t) * μ₀) * (1 - Eq.σ(sqrt(t * (1 - t) * a) * y / μ₀ + (1 - t) * μ₀)) * (1 - t)^2), [0, 1])

        println("μ₀: $μ₀")
        println("Aμ: $Aμ")
        plot!(plt, df_a.x, df_a.μ, label="α = $(a)", color=colors[i], lw=1.5)
        plot!(plt, df_a.x, μ₀ .+ sqrt(a) ./ (μ₀ .* (1 .+ (Aμ ./ λ))) .* df_a.x, label="", color=colors[i], lw=1.5, linestyle=:dash)
    end
    display(plt)
end

function plot_q_and_dq_vs_alpha(df; outdir="../plots", vals=[0.08, 0.1, 0.2, 0.5, 1, 2, 5, 10, 13], exp_csv_path=nothing, L=1024)
    λs = sort(unique(df.λ))
    colors = plot_colors(length(λs))

    # Work on a copy so the caller's dataframe is not mutated.
    df_plot = filter(row -> row.α >= first(vals), copy(df))

    df_exp = nothing
    if exp_csv_path !== nothing
        df_exp = filter(row -> row.dataset == "uniform" && row.L == L, CSV.read(exp_csv_path, DataFrame))
    end

    base = "replica_lambdas_$(fmt_plot_value(first(λs)))_to_$(fmt_plot_value(last(λs)))_alpha_$(fmt_plot_value(first(vals)))_to_$(fmt_plot_value(last(vals)))"

    p = plot(
        title="q and Loss vs α, λ=$(first(λs))",
        xlabel="α",
        ylabel="q",
        xscale=:log10,
        legend=:right,
        xticks=(vals, string.(vals)),
        xlim=[first(vals), last(vals)],
    )
    p_loss = twinx()
    plot!(p_loss, xscale=:log10,
        xticks=(vals, string.(vals)),
        xlim=[first(vals), last(vals)],
        legend=:right,
        ylabel="Train Loss",)

    exp_qw_col = Symbol("train/qW")
    exp_loss_col = Symbol("train/loss")

    for (i, λ) in enumerate(λs)
        df_λ = filter(row -> row.λ == λ, df_plot)
        sort!(df_λ, :α)

        if df_exp !== nothing &&
           "L" in names(df_exp) &&
           "l2reg" in names(df_exp) &&
           "alpha" in names(df_exp) &&
           string(exp_loss_col) in names(df_exp)

            # 1) all experiments with matching l2reg for current λ
            df_exp_λ = filter(
                row -> !ismissing(row.l2reg) &&
                    isapprox(Float64(row.l2reg), Float64(λ); atol=1e-12),
                df_exp,
            )

            # 2) keep rows with valid alpha, L, and loss
            df_exp_λ = filter(
                row -> !ismissing(row.alpha) &&
                           !ismissing(row.L) &&
                           !ismissing(row[exp_qw_col]) &&
                           !ismissing(row[exp_loss_col]),
                df_exp_λ,
            )


            if nrow(df_exp_λ) > 0
                grouped = groupby(df_exp_λ, :alpha)
                α_exp = Float64[]
                qW_exp = Float64[]
                loss_exp = Float64[]

                for g in grouped
                    push!(α_exp, Float64(g.alpha[1]))
                    push!(qW_exp, Float64(mean(g[!, exp_qw_col])))
                    push!(loss_exp, Float64(mean(g[!, exp_loss_col])))
                end

                perm = sortperm(α_exp)
                α_exp = α_exp[perm]
                qW_exp = qW_exp[perm]
                loss_exp = loss_exp[perm]

                scatter!(p, α_exp, qW_exp,
                    label="",
                    marker=:cross,
                    ms=6,
                    color=:lightblue,
                    markerstrokewidth=2,
                    alpha=1
                )
                scatter!(p_loss, α_exp, loss_exp,
                    label="",
                    marker=:cross,
                    ms=6,
                    color=:orange,
                    markerstrokewidth=2,
                    alpha=1
                )
            end
        end


        plot!(p, df_λ.α, df_λ.q, label="q", color=:lightblue, lw=3, alpha=1)
        plot!(p_loss, [], [], label="q", color=:lightblue, lw=3)
        scatter!(p_loss, [], [], marker=:cross, ms=4, color=:lightblue, markerstrokewidth=2, label="q (exp)")
        plot!(p_loss, df_λ.α, .-(df_λ.λ .* df_λ.q ./ 2. .+ df_λ.ϕ) ./ df_λ.α, label="Train Loss", color=:orange, lw=3, alpha=1)
        scatter!(p_loss, [], [], marker=:cross, ms=4, color=:orange, markerstrokewidth=2, label="Train Loss (exp)")
    end

    save_plot(p, outdir, base * "_q_and_loss.png")
    return p
end

function plot_oneshot_accuracy(filename; exp_filename=nothing, xlim=[0, 3], outdir="../plots")
    if exp_filename != nothing
        df_exp = CSV.read(exp_filename, DataFrame)
    end

    df = file_to_df(filename)
    ts = sort(unique(df.t))
    colors = plot_colors(length(ts))

    is_hebbian = occursin("hebbian", lowercase(filename))
    algorithm = match(r"ε_([^_]+)_t", filename).captures[1]

    title_label = is_hebbian ? "one-shot, hebbian, $algorithm algorithm" : "one-shot, λ=$(first(df.λ)), $algorithm algorithm"
    p = plot(title=title_label, xlabel="α", ylabel="m_f", legend=:topright, dpi=600, xlim=xlim)

    for (i, tv) in enumerate(ts)
        if exp_filename != nothing
            df_exp_t0 = filter(row -> row.decoding == algorithm, filter(row -> row.t0 ≈ tv, df_exp))
            plot!(df_exp_t0.alpha, df_exp_t0.frac_correct - df_exp_t0.frac_errors, label="", lw=1, color=:gray, ls=:dash)
            scatter!(df_exp_t0.alpha, df_exp_t0.frac_correct - df_exp_t0.frac_errors, label="", lw=2, color=colors[i], marker=:dtriangle, markerstrokecolor=colors[i])
        end
        df_greedy_t = filter(row -> row.t == tv, df)
        sort!(df_greedy_t, :α)
        plot!(p, df_greedy_t.α, 1 .- 2 * (tv) * (1 .- df_greedy_t.ε), label="t=$(tv)", color=colors[i], lw=2)
    end

    λ_label = is_hebbian ? "hebbian" : fmt_plot_value(first(df.λ))
    outfile = joinpath(
        outdir,
        "oneshot_accuracy_$(λ_label)_alg_$(algorithm)_x_$(fmt_plot_value(xlim[1]))_$(fmt_plot_value(xlim[2]))_nt_$(length(ts)).png",
    )
    save_plot(p, outdir, basename(outfile))

    return p
end

function plot_fixed_alpha_three_cases(df_lambda0, df_lambda001; t_v=0.5, algorithm=:greedy, atol=1e-8, xlim=[0, 3], outdir="../plots")
    @assert algorithm in (:greedy, :fair) "algorithm must be :greedy or :fair"

    # Select rows at fixed t and order by alpha.
    df0 = sort(filter(row -> isapprox(row.t, t_v; atol=atol), df_lambda0), :α)
    df1 = sort(filter(row -> isapprox(row.t, t_v; atol=atol), df_lambda001), :α)

    @assert nrow(df0) > 0 "No rows found in df_lambda0 for t=$(t_v)"
    @assert nrow(df1) > 0 "No rows found in df_lambda001 for t=$(t_v)"

    α0 = df0.α
    α1 = df1.α

    # Experimental curves from datasets.
    m0 = 1 .- 2 .* t_v .* (1 .- df0.ε)
    m1 = 1 .- 2 .* t_v .* (1 .- df1.ε)

    # Hebbian theory evaluated vs alpha at fixed t.
    αmin = min(minimum(α0), minimum(α1))
    αmax = max(maximum(α0), maximum(α1))
    α_hebb = range(αmin, αmax; length=300)
    sigma(x) = 1 / (1 + exp(-x))

    y_hebb = if algorithm == :greedy
        0.5 .* (1 .+ erf.(sqrt.((1 .- t_v) ./ (2 .* α_hebb))))
    else
        [∫D(z -> sigma((1 - t_v) + sqrt(α * (1 - t_v)) * z)) for α in α_hebb]
    end

    m_hebb = 1 .- 2 .* t_v .* (1 .- y_hebb)

    p = plot(
        title="Fixed t=$(t_v), algorithm=$(algorithm)",
        xlabel="α",
        ylabel="m_f",
        legend=:topright,
        dpi=600,
        xlim=xlim,
    )

    # Distinct visual styles for the three cases.
    plot!(p, α0, m0; label="λ=0 data", color=:steelblue, lw=2.5, linestyle=:solid)
    plot!(p, α1, m1; label="λ=0.01 data", color=:crimson, lw=2.5, linestyle=:dash)
    plot!(p, α_hebb, m_hebb; label="Hebbian $(algorithm)", color=:black, lw=2.5, linestyle=:dot)

    outfile = joinpath(
        outdir,
        "fixed_t_three_cases_alg_$(algorithm)_t_$(fmt_plot_value(t_v))_x_$(fmt_plot_value(xlim[1]))_$(fmt_plot_value(xlim[2]))_alpha_$(fmt_plot_value(αmin))_$(fmt_plot_value(αmax)).png",
    )
    save_plot(p, outdir, basename(outfile))

    return p
end

function plot_m_i_vs_m_f(filename; exp_filename=nothing, outdir="../plots")
    df = file_to_df(filename)
    is_hebbian = occursin("hebbian", lowercase(filename))

    if exp_filename != nothing
        df_exp = CSV.read(exp_filename, DataFrame)
        αs = sort(unique(df_exp.alpha))
    else
        αs = sort(unique(df.α))
    end

    m_alg = match(r"ε_([^_]+)_t", filename)
    algorithm = m_alg === nothing ? "unknown" : m_alg.captures[1]
    λ_label = is_hebbian ? "hebbian" : "λ=$(first(df.λ))"

    p = plot(
        title="one-shot, $(λ_label), $(algorithm) algorithm",
        xlabel="m_i",
        ylabel="m_f",
        legend=:outerright,
        dpi=600,
    )

    colors = plot_colors(length(αs), :dense)

    for (i, αv) in enumerate(αs)
        if αv == 0
            continue
        end

        df_α = filter(row -> row.α == αv, df)
        sort!(df_α, :t)

        if nrow(df_α) == 0
            continue
        end

        m_i = 1 .- df_α.t
        m_f = 1 .- 2 .* df_α.t .* (1 .- df_α.ε)

        if exp_filename != nothing
            df_exp_α = filter(row -> row.alpha == αv, df_exp)

            if nrow(df_exp_α) > 0 && "frac_correct" in names(df_exp_α) && "frac_errors" in names(df_exp_α)
                m_i_exp = 1 .- df_exp_α.t0
                m_f_exp = df_exp_α.frac_correct .- df_exp_α.frac_errors
                plot!(p, m_i_exp, m_f_exp, label="", color=:gray, lw=1, ls=:dash)
                scatter!(p, m_i_exp, m_f_exp, label="", marker=:circle, color=colors[i], ms=4, markerstrokewidth=0)
            end
        end

        plot!(p, m_i, m_f; lw=2, ms=3, color=colors[i], label="α=$(round(αv, digits=1))")

    end

    outfile = joinpath(
        outdir,
        "m_i_vs_m_f_$(λ_label)_alg_$(algorithm)_alpha_$(fmt_plot_value(first(αs)))_$(fmt_plot_value(last(αs))).png",
    )
    save_plot(p, outdir, basename(outfile))

    return p
end

function plot_integrated_accuracy(filename; exp_filename=nothing, xlim=[0, 3], outdir="../plots")
    if exp_filename != nothing
        df_exp = CSV.read(exp_filename, DataFrame)
    end

    is_hebbian = occursin("hebbian", lowercase(filename))

    df = file_to_df(filename)
    ts = sort(unique(df.t_0))
    colors = plot_colors(length(ts))
    if occursin("greedy", lowercase(filename))
        algorithm = "greedy"
    elseif occursin("fair", lowercase(filename))
        algorithm = "fair"
    else
        algorithm = "unknown"
    end
    λ_label = is_hebbian ? "hebbian" : "λ=$(first(df.λ))"

    p = plot(title="integrated, $λ_label, $algorithm algorithm", xlabel="α", ylabel="m_f", legend=:topright, dpi=600, xlim=xlim)

    for (i, tv) in enumerate(ts)
        if exp_filename != nothing
            df_exp_t0 = filter(row -> row.decoding == algorithm, filter(row -> row.frac_masked == 0.0, filter(row -> row.t0 ≈ tv, df_exp)))
            plot!(df_exp_t0.alpha, df_exp_t0.frac_correct - df_exp_t0.frac_errors, label="", lw=1, color=:gray, ls=:dash)
            scatter!(df_exp_t0.alpha, df_exp_t0.frac_correct - df_exp_t0.frac_errors, label="", lw=2, color=colors[i], marker=:dtriangle, markerstrokecolor=colors[i])
        end
        df_greedy_t = filter(row -> row.t_0 == tv, df)
        sort!(df_greedy_t, :α)
        plot!(p, df_greedy_t.α, df_greedy_t.m_f, label="t=$(tv)", color=colors[i], lw=2)
    end

    λv = first(df.λ)
    outfile = joinpath(
        outdir,
        "integrated_accuracy_lambda_$(is_hebbian ? "hebbian" : fmt_plot_value(λv))_alg_$(algorithm)_x_$(fmt_plot_value(xlim[1]))_$(fmt_plot_value(xlim[2]))_nt_$(length(ts)).png",
    )
    save_plot(p, outdir, basename(outfile))

    return p
end

function plot_integrated_m_i_vs_m_f(filename; exp_filename=nothing, outdir="../plots")
    df = file_to_df(filename)
    is_hebbian = occursin("hebbian", lowercase(filename))

    if exp_filename != nothing
        df_exp = CSV.read(exp_filename, DataFrame)
        αs = sort(unique(df_exp.alpha))
    else
        αs = sort(unique(df.α))
    end

    m_alg = match(r"t0_([^_]+)\.txt$", filename)
    algorithm = m_alg === nothing ? "unknown" : m_alg.captures[1]
    λ_label = is_hebbian ? "hebbian" : "λ=$(first(df.λ))"

    p = plot(
        title="integrated, $(λ_label), $(algorithm) algorithm",
        xlabel="m_i",
        ylabel="m_f",
        legend=:outerright,
        dpi=600,
    )

    colors = plot_colors(length(αs), :dense)

    for (i, αv) in enumerate(αs)
        if αv == 0
            continue
        end

        df_α = filter(row -> row.α == αv, df)
        sort!(df_α, :t_0)

        if nrow(df_α) == 0
            continue
        end

        m_i = 1 .- df_α.t_0
        m_f = df_α.m_f

        if exp_filename != nothing
            if "frac_masked" in names(df_exp)
                df_exp_α = filter(row -> row.alpha == αv && row.frac_masked == 0.0, df_exp)
            else
                df_exp_α = filter(row -> row.alpha == αv, df_exp)
            end

            if nrow(df_exp_α) > 0 && "frac_correct" in names(df_exp_α) && "frac_errors" in names(df_exp_α)
                m_i_exp = 1 .- df_exp_α.t0
                m_f_exp = df_exp_α.frac_correct .- df_exp_α.frac_errors
                plot!(p, m_i_exp, m_f_exp, label="", color=:gray, lw=1, ls=:dash)
                scatter!(p, m_i_exp, m_f_exp, label="", marker=:circle, color=colors[i], ms=4, markerstrokewidth=0)
            end
        end

        plot!(p, m_i, m_f; lw=2, ms=3, color=colors[i], label="α=$(round(αv, digits=1))")
    end

    outfile = joinpath(
        outdir,
        "integrated_m_i_vs_m_f_$(λ_label)_alg_$(algorithm)_alpha_$(fmt_plot_value(first(αs)))_$(fmt_plot_value(last(αs))).png",
    )
    save_plot(p, outdir, basename(outfile))

    return p
end

function plot_exp_oneshot_vs_integrated(exp_filename_oneshot, exp_filename_integrated; xlim=[0, 3], t_values=0:0.2:1, outdir="../plots", algorithm="greedy")
    df_exp_oneshot = CSV.read(exp_filename_oneshot, DataFrame)
    df_exp_integrated = CSV.read(exp_filename_integrated, DataFrame)

    ts = sort(t_values)
    colors = plot_colors(length(ts))

    λ = match(r"l2reg([0-9.]+)", exp_filename_oneshot).captures[1]

    p = plot(title="λ=$λ, $algorithm algorithm", xlabel="α", ylabel="m_f", legend=:topright, dpi=600, xlim=xlim)

    for (i, tv) in enumerate(ts)
        df_exp_oneshot_t0 = filter(row -> row.decoding == algorithm, filter(row -> row.t0 ≈ tv, df_exp_oneshot))
        plot!(df_exp_oneshot_t0.alpha, df_exp_oneshot_t0.frac_correct - df_exp_oneshot_t0.frac_errors, label="$(i==1 ? "one shot" : "")", lw=2, color=colors[i], ls=:dash)
        scatter!(df_exp_oneshot_t0.alpha, df_exp_oneshot_t0.frac_correct - df_exp_oneshot_t0.frac_errors, marker=:circle, color=colors[i], ms=4, markerstrokewidth=0, label="")


        df_exp_integrated_t0 = filter(row -> row.decoding == algorithm, filter(row -> row.frac_masked == 0.0, filter(row -> row.t0 ≈ tv, df_exp_integrated)))
        plot!(df_exp_integrated_t0.alpha, df_exp_integrated_t0.frac_correct - df_exp_integrated_t0.frac_errors, label="$(i==1 ? "integrated" : "")", lw=2, color=colors[i],)
        scatter!(df_exp_integrated_t0.alpha, df_exp_integrated_t0.frac_correct - df_exp_integrated_t0.frac_errors, marker=:circle, color=colors[i], ms=4, markerstrokewidth=0, label="")


    end

    fmt(x) = replace(string(round(x, digits=4)), "." => "p")
    outfile = joinpath(
        outdir,
        "oneshot_vs_integrated_λ_$(λ)_alg_$(algorithm)_x_$(fmt_plot_value(xlim[1]))_$(fmt_plot_value(xlim[2]))_nt_$(length(ts)).png",
    )
    save_plot(p, outdir, basename(outfile))

    return p
end

function plot_oneshot_vs_integrated(filename_oneshot, filename_integrated; xlim=[0, 3], t_values=0:0.2:1, outdir="../plots", algorithm="greedy")
    df_oneshot = file_to_df(filename_oneshot)
    df_integrated = file_to_df(filename_integrated)

    ts = sort(t_values)
    colors = plot_colors(length(ts))

    λ = first(df_oneshot.λ)

    p = plot(title="λ=$λ, $algorithm algorithm", xlabel="α", ylabel="m_f", legend=:topright, dpi=600, xlim=xlim)

    for (i, tv) in enumerate(ts)


        df_oneshot_t0 = filter(row -> row.t == tv, df_oneshot)
        sort!(df_oneshot_t0, :α)
        plot!(df_oneshot_t0.α, 1 .- 2 * (tv) * (1 .- df_oneshot_t0.ε), label="$(i==1 ? "one shot" : "")", lw=2, color=colors[i], ls=:dash)

        df_integrated_t0 = filter(row -> row.t_0 == tv, df_integrated)
        sort!(df_integrated_t0, :α)
        plot!(df_integrated_t0.α, df_integrated_t0.m_f, label="$(i==1 ? "integrated" : "")", lw=2, color=colors[i],)
    end

    outfile = joinpath(
        outdir,
        "oneshot_vs_integrated_λ_$(fmt_plot_value(λ))_alg_$(algorithm)_x_$(fmt_plot_value(xlim[1]))_$(fmt_plot_value(xlim[2]))_nt_$(length(ts)).png",
    )
    save_plot(p, outdir, basename(outfile))

    return p
end

function plot_exp_oneshot_vs_integrated_mi_mf(exp_filename_oneshot, exp_filename_integrated; alpha_values=nothing, outdir="../plots", algorithm="greedy", xlim=[0, 1], ylim=[0, 1])
    df_exp_oneshot = CSV.read(exp_filename_oneshot, DataFrame)
    df_exp_integrated = CSV.read(exp_filename_integrated, DataFrame)

    αs = if alpha_values === nothing
        sort(unique(df_exp_oneshot.alpha))
    else
        sort(alpha_values)
    end

    n_α = length(αs)
    colors = plot_colors(n_α, :dense)

    λ_match = match(r"l2reg([0-9.eE+-]+)", exp_filename_oneshot)
    λ_label = λ_match === nothing ? "unknown" : λ_match.captures[1]

    p = plot(
        title="λ=$λ_label, $algorithm algorithm",
        xlabel="m_i",
        ylabel="m_f",
        legend=:outerright,
        dpi=600,
        xlim=xlim,
        ylim=ylim,
    )

    for (i, αv) in enumerate(αs)
        df_oneshot_α = filter(
            row -> !ismissing(row.alpha) && row.decoding == algorithm && row.alpha ≈ αv,
            df_exp_oneshot,
        )
        sort!(df_oneshot_α, :t0)

        if nrow(df_oneshot_α) > 0
            m_i_oneshot = 1 .- df_oneshot_α.t0
            m_f_oneshot = df_oneshot_α.frac_correct .- df_oneshot_α.frac_errors

            plot!(p, m_i_oneshot, m_f_oneshot,
                label="",
                lw=2,
                color=colors[i],
                ls=:dash,
            )
        end

        df_integrated_α = if "frac_masked" in names(df_exp_integrated)
            filter(
                row -> !ismissing(row.alpha) && row.decoding == algorithm && row.frac_masked == 0.0 && row.alpha ≈ αv,
                df_exp_integrated,
            )
        else
            filter(
                row -> !ismissing(row.alpha) && row.decoding == algorithm && row.alpha ≈ αv,
                df_exp_integrated,
            )
        end
        sort!(df_integrated_α, :t0)

        if nrow(df_integrated_α) > 0
            m_i_integrated = 1 .- df_integrated_α.t0
            m_f_integrated = df_integrated_α.frac_correct .- df_integrated_α.frac_errors

            plot!(p, m_i_integrated, m_f_integrated,
                label=αv,
                lw=2,
                color=colors[i],
            )
        end
    end

    outfile = joinpath(outdir, "oneshot_vs_integrated_mi_vs_mf_λ_$(λ_label)_alg_$(algorithm)_nα_$(n_α).png")
    save_plot(p, outdir, basename(outfile))

    return p
end

function plot_trajectories_m_vs_t(filename; exp_filename=nothing, outdir="../plots", α_filter=nothing)
    """
    Plot m trajectories vs t for each t_0, with multiple lines colored by t_0.
    Optionally overlay experimental scatter points.

    Parameters:
    - filename: path to trajectory data file with columns α, λ, q, δq, t_0, m_t0, m_f, t, m
    - exp_filename: (optional) path to experimental CSV with columns alpha, frac_masked, frac_correct, frac_errors
    - outdir: output directory for the plot
    - α_filter: (optional) filter to rows matching specific alpha value(s)
    """
    df = file_to_df(filename)
    if exp_filename !== nothing
        df_exp = CSV.read(exp_filename, DataFrame)
    end

    # Filter by α if specified
    if α_filter !== nothing
        if !(α_filter isa AbstractVector) && !(α_filter isa AbstractRange)
            α_filter = [α_filter]
        end
        df = filter(row -> row.α in α_filter, df)
    end

    if nrow(df) == 0
        error("No rows matching filter in $filename")
    end

    # Get unique parameters for labeling
    αs = unique(df.α)
    n_α = length(αs)
    is_hebbian = occursin("hebbian", lowercase(filename))
    λ_label = is_hebbian ? "hebbian" : "λ=$(first(df.λ))"
    algorithm = match(r"(fair|greedy)", filename).captures[1]

    # Check if this is the first α (for title)
    α_val = first(αs)

    # Get unique t_0 values and sort in descending order (for legend clarity)
    t_0_values = sort(unique(df.t_0), rev=true)
    colors = plot_colors(length(t_0_values), :viridis)

    p = plot(
        title="trajectory for α=$α_val, $λ_label, $algorithm algorithm",
        xlabel="t",
        ylabel="m(t)",
        legend=:topright,
        dpi=600,
        xlim=[0, maximum(df.t_0)],
        ylim=[minimum(df.m) - 0.01, maximum(df.m) + 0.01],
    )

    # Plot one line per t_0
    for (idx, t_0) in enumerate(t_0_values)
        if t_0 == 0 || t_0 == 1.0
            continue  # Skip t_0=0 if it exists, as it may be trivial
        end
        df_t0 = filter(row -> isapprox(row.t_0, t_0; atol=1e-10), df)
        sort!(df_t0, :t)

        if exp_filename !== nothing
            df_exp_t0 = filter(row -> row.t0 ≈ t_0 && row.alpha == α_val, df_exp)
            plot!(p, df_exp_t0.frac_masked, df_exp_t0.frac_correct;
                label="t₀=$t_0",
                lw=2.3,
                color=colors[idx],
                alpha=1.)
            plot!(p, df_exp_t0.frac_masked, df_exp_t0.frac_errors;
                label="",
                lw=2.3,
                color=colors[idx],
                alpha=1.)
        end
        if nrow(df_t0) > 0
            plot!(p, df_t0.t, (1 .- df_t0.t .+ df_t0.m) ./ 2;
                lw=2.3,
                label="",
                color=colors[idx],
                alpha=1.,
                ls=:dot)
            plot!(p, df_t0.t, (1 .- df_t0.t .- df_t0.m) ./ 2;
                lw=2.3,
                label="",
                color=colors[idx],
                alpha=1.,
                ls=:dot)
        end
    end


    outfile = joinpath(outdir, "trajectories_m_vs_t_alpha_$(fmt_plot_value(α_val))_$(λ_label)_$algorithm.png")
    save_plot(p, outdir, basename(outfile))

    println("Plot saved to: $outfile")
    return p
end

function plot_trajectories_v_vs_t(filename; exp_filename=nothing, outdir="../plots", α_filter=nothing)
    """
    Plot v trajectories vs t for each t_0, with multiple lines colored by t_0.
    Optionally overlay experimental scatter points.

    Parameters:
    - filename: path to trajectory data file with columns α, λ, q, δq, t_0, m_t0, m_f, t, m
    - exp_filename: (optional) path to experimental CSV with columns alpha, frac_masked, frac_correct, frac_errors
    - outdir: output directory for the plot
    - α_filter: (optional) filter to rows matching specific alpha value(s)
    """
    df = file_to_df(filename)
    if exp_filename !== nothing
        df_exp = CSV.read(exp_filename, DataFrame)
    end

    # Filter by α if specified
    if α_filter !== nothing
        if !(α_filter isa AbstractVector) && !(α_filter isa AbstractRange)
            α_filter = [α_filter]
        end
        df = filter(row -> row.α in α_filter, df)
    end

    if nrow(df) == 0
        error("No rows matching filter in $filename")
    end

    # Get unique parameters for labeling
    αs = unique(df.α)
    α_val = first(αs)
    is_hebbian = occursin("hebbian", lowercase(filename))
    λ_label = is_hebbian ? "hebbian" : "λ=$(first(df.λ))"
    algorithm = match(r"(fair|greedy)", filename).captures[1]
    t_0_values = sort(unique(df.t_0), rev=true)

    # Create color palette for different t_0 values
    colors = plot_colors(length(t_0_values), :viridis)

    p = plot(
        title="trajectory for α=$α_val, $λ_label, $algorithm algorithm",
        xlabel="t",
        ylabel="v(t)",
        legend=:topright,
        dpi=600,
        xlim=[0, maximum(df.t_0)],
        ylim=[minimum(df.v) - 0.01, maximum(df.v) + 0.01],
    )

    # Plot one line per t_0
    for (idx, t_0) in enumerate(t_0_values)
        if t_0 == 0 || t_0 == 1.0
            continue  # Skip t_0=0 if it exists, as it may be trivial
        end
        df_t0 = filter(row -> isapprox(row.t_0, t_0; atol=1e-10), df)
        sort!(df_t0, :t)


        if nrow(df_t0) > 0
            plot!(p, df_t0.t, df_t0.v;
                lw=2.3,
                label="",
                color=colors[idx],
                alpha=1.,)
            plot!(p, df_t0.t, 1 .- df_t0.t;
                lw=2.3,
                label="",
                color=colors[idx],
                alpha=1.,
                ls=:dot)
        end
    end


    outfile = joinpath(outdir, "trajectories_v_vs_t_alpha_$(fmt_plot_value(α_val))_$(λ_label)_$algorithm.png")
    save_plot(p, outdir, basename(outfile))

    println("Plot saved to: $outfile")
    return p
end

function plot_trajectories_derivative_m_vs_t(filename; exp_filename=nothing, outdir="../plots", α_filter=nothing)
    """
    Plot m trajectories vs t for each t_0, with multiple lines colored by t_0.
    Optionally overlay experimental scatter points.

    Parameters:
    - filename: path to trajectory data file with columns α, λ, q, δq, t_0, m_t0, m_f, t, m
    - exp_filename: (optional) path to experimental CSV with columns alpha, frac_masked, frac_correct, frac_errors
    - outdir: output directory for the plot
    - α_filter: (optional) filter to rows matching specific alpha value(s)
    """
    df = file_to_df(filename)
    if exp_filename !== nothing
        df_exp = CSV.read(exp_filename, DataFrame)
    end

    # Filter by α if specified
    if α_filter !== nothing
        if !(α_filter isa AbstractVector) && !(α_filter isa AbstractRange)
            α_filter = [α_filter]
        end
        df = filter(row -> row.α in α_filter, df)
    end

    if nrow(df) == 0
        error("No rows matching filter in $filename")
    end

    # Get unique parameters for labeling
    αs = unique(df.α)
    is_hebbian = occursin("hebbian", lowercase(filename))
    λ_label = is_hebbian ? "hebbian" : "λ=$(first(df.λ))"
    algorithm = match(r"t0_([^_]+)_trajectory", filename).captures[1]


    # Check if this is the first α (for title)
    df_a = sorted_subset(df, row -> row.α == a, :x)

    # Get unique t_0 values and sort in descending order (for legend clarity)
    t_0_values = sort(unique(df.t_0), rev=true)

    # Create color palette for different t_0 values
    colors = plot_colors(length(t_0_values))

    p = plot(
        title="trajectory for α=$α_val, $λ_label, $algorithm algorithm",
        xlabel="t",
        ylabel="dm / dt",
        legend=:bottomright,
        dpi=600,
        # xlim=[0, maximum(df.t_0)],
        # ylim=[minimum(df.m) - 0.01, maximum(df.m) + 0.01],
    )

    # Plot one line per t_0
    for (idx, t_0) in enumerate(t_0_values)
        df_t0 = filter(row -> isapprox(row.t_0, t_0; atol=1e-10), df)
        sort!(df_t0, :t)



        if exp_filename !== nothing
            df_exp_t0 = filter(row -> row.t0 ≈ t_0 && row.alpha == α_val, df_exp)
            print(nrow(df_exp_t0))
            t_mid = (df_exp_t0.frac_masked[1:end-1] .+ df_exp_t0.frac_masked[2:end]) ./ 2
            dm_dt = diff(df_exp_t0.frac_correct .- df_exp_t0.frac_errors) ./ diff(df_exp_t0.frac_masked)

            plot!(p, t_mid, dm_dt;
                label="",
                color=colors[idx],
                lw=2.3,
                alpha=0.5
            )
        end
        if nrow(df_t0) > 0
            t_mid = (df_t0.t[1:end-1] .+ df_t0.t[2:end]) ./ 2
            dm_dt = diff(df_t0.m) ./ diff(df_t0.t)

            plot!(p, t_mid, dm_dt;
                label="t₀=$(t_0)",
                color=colors[idx],
                lw=2.3,
            )
        end
    end


    outfile = joinpath(outdir, "trajectories_m_vs_t_alpha_$(fmt_plot_value(α_val))_$(λ_label)_$algorithm.png")
    save_plot(p, outdir, basename(outfile))

    println("Plot saved to: $outfile")
    return p
end

function dataset_size_percent_from_model(model::AbstractString)
    match_result = match(r"mdlm_(0*)([1-9])_", model)
    match_result === nothing && error("Could not extract dataset size from model: $model")

    zero_count = length(match_result.captures[1])
    x = parse(Float64, match_result.captures[2])
    return x * 100 / 10^zero_count
end

function plot_f_mem_by_dataset_size(csv_path::AbstractString; outdir="../plots",)
    df = CSV.read(csv_path, DataFrame)
    df.dataset_size_percent = dataset_size_percent_from_model.(df.model)
    sort!(df, :dataset_size_percent)

    ham_cols = [:f_mem_ham_600, :f_mem_ham_700, :f_mem_ham_800, :f_mem_ham_900]
    ham_percents = [60, 70, 80, 90]
    ham_labels = String.(ham_cols)

    ham_gradient = plot_colors(length(ham_labels) + 5, :dense)


    p = plot(
        xlabel="dataset size (%)",
        ylabel="f_mem",
        xscale=:log10,
        xticks=([1, 10, 100], ["1", "10", "100"]),
        legend=:topright,
        grid=true,
        title="f_mem vs dataset size",
    )

    plot!(
        p,
        df.dataset_size_percent,
        df.f_mem,
        label="L2",
        color=:tomato,
        linewidth=3,
        marker=:circle,
        markerstrokecolor=:tomato,
        markersize=5,
    )

    for (i, col) in enumerate(ham_cols)
        plot!(
            p,
            df.dataset_size_percent,
            df[!, col],
            label="Hamming ($(100-ham_percents[i])%)",
            color=ham_gradient[i],
            linewidth=3,
            marker=:circle,
            markerstrokecolor=ham_gradient[i],
            markersize=5,
        )
    end

    outfile = joinpath(outdir, "f_mem_by_dataset_size.png")
    save_plot(p, outdir, basename(outfile))

    println("Plot saved to: $outfile")

    return p
end
