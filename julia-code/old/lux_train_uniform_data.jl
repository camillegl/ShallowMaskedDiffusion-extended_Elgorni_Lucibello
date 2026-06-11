using Random, Statistics, LinearAlgebra
using Optimisers: Optimsers, AdamW
using Lux
using Zygote
using Enzyme
using MLUtils: DataLoader
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using ComponentArrays: ComponentVector
using OneHotArrays: onecold


# x_i ∈ {-1,0,1}. 0 represents the mask

####### MODEL DEFINITION ##########

struct DiffusionBackbone{F1, F2} <: LuxCore.AbstractLuxLayer
    L::Int
    init_weight::F1
    init_bias::F2
end

function DiffusionBackbone(L::Int; init_weight=glorot_uniform, init_bias=zeros32)
    return DiffusionBackbone(L, init_weight, init_bias)
end


LuxCore.initialparameters(rng::AbstractRNG, m::DiffusionBackbone) =
    (W = m.init_weight(rng, m.L, m.L), V = m.init_weight(rng, m.L, m.L))

LuxCore.initialstates(rng::AbstractRNG, m::DiffusionBackbone) = (;)
LuxCore.parameterlength(m::DiffusionBackbone) = 2 * m.L * m.L
LuxCore.statelength(m::DiffusionBackbone) = 0


function (m::DiffusionBackbone)(x::AbstractMatrix, ps, st)
    return ps.W * x .+ ps.V * (x .== 0), st
end

function (m::DiffusionBackbone)(x::AbstractArray, ps, st)
    if ndims(x) > 2
        L, batch_size, rest... = size(x)
        x = reshape(x, L, batch_size * prod(rest))
        y, st = m(x, ps, st)
        y = reshape(y, L, batch_size, rest...)
        return y, st
    else
        return m(x, ps, st)
    end
end

function loss_fn(model, ps, st, x0)
    L, batch_size = size(x0)
    t = rand(Float32, 1, batch_size)
    t = repeat(t, L, 1)
    mask = rand(Float32, L, batch_size) .< t
    xt = mask .* 0 .+ .!mask .* x0
    logits, st = model(xt, ps, st)
    y = @. (x0 + 1) ÷ 2
    losses = BinaryCrossEntropyLoss(logits=true, agg=identity)(logits[mask], y[mask])
    @assert length(losses) == sum(mask)
    weight = 1 ./ t[mask] # TODO add epsilon?
    loss = sum(losses .* weight) / (L * batch_size)
    return loss, st, nothing
end

function compute_accuracy(smodel::StatefulLuxLayer, x0; nsamples=10)
    total_correct, total = 0, 0
    for _ in 1:nsamples
        t = rand(Float32, 1, batch_size)
        mask = rand(Float32, L, batch_size) .< t
        xt = mask .* 0 .+ .!mask .* x0
        logits = smodel(xt)
        total_correct += sum((logits[mask] .* x0[mask]) .> 0)
        total += sum(mask)
    end
    return total_correct / total
end
##############################

function generate_dataset(rng::AbstractRNG, L::Int, α::Float64)
    M = round(Int, α * L)
    x = rand(rng, [-1, 1], L, M)
    return x
end

###############################
device = cpu_device()
L = 128
α = 0.1
M = round(Int, α * L)
lambda = 1f-6 / M
batch_size = M
epochs = 1000
ad = AutoZygote() # or AutoEnzyme()

rng = Random.default_rng()
Random.seed!(rng, 42)

train_dataset = generate_dataset(rng, L, α)
train_dataloader = DataLoader(train_dataset; batchsize=batch_size, shuffle=true)


opt = Optimisers.AdamW(eta=1e-3, lambda=lambda)
model = DiffusionBackbone(L)
ps, st = LuxCore.setup(rng, model) |> device
tstate = Training.TrainState(model, ps, st, opt)
smodel = StatefulLuxLayer(model, tstate.parameters, tstate.states)

avg_loss = 1.0
for epoch in 1:epochs
    for x0 in train_dataloader
        x0 = device(x0)
        _, loss, _, tstate = Training.single_train_step!(ad, loss_fn, x0, tstate)
        avg_loss = 0.9 * avg_loss + 0.1 * loss
    end

    acc = compute_accuracy(smodel, train_dataset, nsamples=10)
    println("Epoch $epoch, Loss: $avg_loss, Accuracy: $acc")
end




### Attempt with LBFGS, but since the loss is stochastic, it does not work well

# ps, st = LuxCore.setup(rng, model) |> device
# ps = ComponentArray(ps)

# function callback(state, l)
#     println("Iteration: $(state.iter), Loss: $l")
#     return false
# end

# opt_func = OptimizationFunction(ad) do ps, _
#     loss_data = loss_fn(model, ps, st, train_dataset)[1]
#     loss_reg = lambda * sum(ps.^2) / 2
#     return loss_data #+ loss_reg
# end

# opt_prob = OptimizationProblem(opt_func, ps |> f64)
# # res = Optimization.solve(opt_prob, Optim.LBFGS(); callback)

# res = Optimization.solve(opt_prob, Optimization.LBFGS(); callback) # cast to f64 for using this

