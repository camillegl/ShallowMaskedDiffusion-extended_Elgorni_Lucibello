# Plot the RS entropy profile of the hidden-manifold sign channel from CSVs
# in data/. Auto-discovers
#   data/entropy_sign_shannon.csv         (Shannon, α = 1)
#   data/entropy_sign_renyi_alpha<α>.csv  (Rényi at order α)
# and overlays each curve on the Cover/Hartley reference s₀ = h₂(1/γ).
#
# Output: plots/entropy_sign_renyi_all.pdf  (or path passed as ARGS[1])
#
# Usage from julia-code/hiddenmanifold/:
#   julia --project=. scripts/plot_entropy.jl
#   julia --project=. scripts/plot_entropy.jl plots/custom.pdf

using DelimitedFiles
using Plots
using Printf

const ROOT     = abspath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(ROOT, "data")
const DEFAULT_OUT = joinpath(ROOT, "plots", "entropy_sign_renyi_all.pdf")

h2(p) = -p*log(p) - (1-p)*log(1-p)
s0(γ) = γ ≤ 2 ? log(2) : h2(1/γ)

function discover_renyi_files(dir)
    entries = []
    # Tolerate the historical "_q0=0" suffix on the filename.
    for f in readdir(dir)
        m = match(r"^entropy_sign_renyi_alpha([0-9.]+?)(_q0=0)?\.csv$", f)
        m === nothing && continue
        α = parse(Float64, m.captures[1])
        push!(entries, (; α, file = f))
    end
    return sort(entries, by = e -> e.α)
end

# Read a CSV that may have leading "#" comment lines and pull out (gamma, s).
function load_curve(path; gamma_col = "gamma", value_col = "s")
    lines = filter(l -> !startswith(l, "#"), readlines(path))
    hdr   = strip.(split(strip(lines[1]), ","))
    rows  = [parse.(Float64, split(strip(l), ","))
             for l in lines[2:end] if !isempty(strip(l))]
    mat   = reduce(hcat, rows)
    γidx  = findfirst(==(gamma_col), hdr)
    sidx  = findfirst(==(value_col), hdr)
    γs    = mat[γidx, :]
    ss    = mat[sidx, :]
    valid = .!isnan.(ss) .& .!isnan.(γs)
    return γs[valid], ss[valid]
end

# Log-scale color across the available Rényi α range.
function α_color(α, αs)
    αmin, αmax = extrema(αs)
    αmin == αmax && return cgrad(:viridis)[0.5]
    t = (log(α) - log(αmin)) / (log(αmax) - log(αmin))
    return cgrad(:viridis)[clamp(t, 0.0, 1.0)]
end

function main()
    out_path = isempty(ARGS) ? DEFAULT_OUT : ARGS[1]

    sh_path  = joinpath(DATA_DIR, "entropy_sign_shannon.csv")
    isfile(sh_path) || error("Shannon CSV not found at $sh_path — run scripts/sweep_shannon.jl first")
    γs_sh, s_sh = load_curve(sh_path)

    entries  = discover_renyi_files(DATA_DIR)
    αs_avail = [e.α for e in entries]

    println("Discovered $(length(entries)) Rényi CSVs:")
    for e in entries
        @printf("  α = %5.2f   file = %s\n", e.α, e.file)
    end

    plt = plot(γs_sh, s0.(γs_sh);
        label        = raw"$\alpha \to 0$  (Cover, $h_2(1/\gamma)$)",
        xlabel       = raw"load $\gamma = N/D$",
        ylabel       = raw"entropy density (nats)",
        title        = "Hidden-manifold sign channel — RS Rényi entropy profile",
        lw           = 1.8, linestyle = :dashdot, color = :gray,
        legend       = :outerright,
        xlims        = (0, maximum(γs_sh)),
        ylims        = (-0.02, log(2) * 1.05),
        framestyle   = :box,
        grid         = true,
        size         = (1000, 620))

    for e in entries
        γs, ss = load_curve(joinpath(DATA_DIR, e.file))
        isempty(γs) && continue
        αstr = @sprintf("%g", e.α)
        plot!(plt, γs, ss;
              label = "\$\\alpha = $αstr\$",
              lw    = 1.8,
              color = α_color(e.α, αs_avail))
    end

    # Shannon last so it sits on top.
    plot!(plt, γs_sh, s_sh;
        label = raw"$\alpha = 1$  (Shannon)",
        lw    = 2.6, color = :black)

    hline!(plt, [log(2)]; label = raw"$\log 2$",
           lw = 1, linestyle = :dot, color = :black)
    vline!(plt, [2.0]; label = raw"$\gamma_c = 2$",
           lw = 1, linestyle = :dot, color = :red)

    savefig(plt, out_path)
    println("\nWrote ", out_path)
end

main()
