using Pkg; Pkg.activate(".")
using Random, NPZ, Statistics
using MatrixFreeNewton

function michalewicz(x::AbstractVector{T}) where T
    f = zero(T)
    for i = 1:size(x)[1]
        f -= sin(x[i])*(sin((i*x[i]^2)/pi))^(20)
    end
    return f
end

n = 100
iterations = 100
lrsfn_rank = Int(floor(0.5*n))

# Allocate logger dictionaries for seeding
loggers_gd = Dict()
loggers_csgd = Dict()
loggers_newton = Dict()
loggers_sfn = Dict()


for seed in 0:10
    random_state = Random.MersenneTwister(seed)
    x_0 = randn(random_state,n)

    println("Now for gradient descent ")
    w_star_gd,logger_gd = gradientDescent(michalewicz,x_0,iterations = iterations)
    loggers_gd[seed] = logger_gd

    println("Now for curvature gradient descent ")
    w_star_gd,logger_csgd = curvatureScaledGradientDescent(michalewicz,x_0,iterations = iterations)
    loggers_csgd[seed] = logger_csgd

    print("Now for Newton \n")
    w_star_newton,logger_newton = fullNewton(michalewicz,x_0,alpha = 1e0,iterations=iterations)
    loggers_newton[seed] = logger_newton

    println("Now for low rank SFNewton with full rank Hessian ")
    w_star_newton,logger_sfn = lowRankSaddleFreeNewton(michalewicz,x_0,printing_frequency=iterations,
                                                                            iterations = iterations,rank = n)
    loggers_sfn[seed] = logger_sfn
end

loggers_lrsfn = Dict()

for lrsfn_rank in 10:10:90
    loggers_lrsfn[lrsfn_rank] = Dict()
    for seed in 1:10
        random_state = Random.MersenneTwister(seed)
        x_0 = randn(random_state,n)
        println("Now for low rank SFNewton with reduced Hessian with LRSFN rank = ",lrsfn_rank)
        w_star_newton,logger_lrsfn = lowRankSaddleFreeNewton(michalewicz,x_0,alpha = 1e0,printing_frequency=iterations,
                                                            gamma = 1e-3,iterations = iterations,rank = lrsfn_rank,
                                                            log_full_spectrum = true)
        loggers_lrsfn[lrsfn_rank][seed] = logger_lrsfn
    end
end
println("Done")

# Save data for post-processing
data_dir = "michalewicz_data/"
if ~isdir(data_dir)
    mkdir(data_dir)
end
problem_name = "michalewicz_d="*string(n)
optimizers = ["gd","csgd","newton","sfn"]
logger_dicts = [loggers_gd,loggers_csgd,loggers_newton,loggers_sfn]

for (optimizer,logger_dict) in zip(optimizers,logger_dicts)
    println("optimizer = ",optimizer)
    opt_losses = zeros(0)
    for (seed,logger) in logger_dict
        name = problem_name*optimizer*"_"*string(seed)
        println("name = ",name)
        # Save losses
        npzwrite(data_dir*name*"_losses.npy",logger.losses)
        min_loss = minimum(logger.losses)
        append!(opt_losses,min_loss)
        # If sfn save spectrum:
        if optimizer == "sfn"
            npzwrite(data_dir*name*"_spectra.npy",logger.spectra)
        end
        # If csgd save alphas
        if optimizer in ["csgd"]
            npzwrite(data_dir*name*"_alphas.npy",logger.alphas)
        end
    end
    println("Min min loss = ",minimum(opt_losses))
    println("Avg min loss = ",Statistics.mean(opt_losses))
    println("Std min loss = ",Statistics.std(opt_losses,corrected = false))
end

optimizer = "lrsfn"
for (lrsfn_rank,rank_logger) in loggers_lrsfn
   opt_losses = zeros(0)
    for (seed,logger) in rank_logger
        name = problem_name*optimizer*"_"*string(seed)
        name *="rank_"*string(logger.rank)
        println("name = ",name)
        npzwrite(data_dir*name*"_losses.npy",logger.losses)
        min_loss = minimum(logger.losses)
        append!(opt_losses,min_loss)
        npzwrite(data_dir*name*"_spectra.npy",logger.spectra)
    end
    println("Min min loss = ",minimum(opt_losses))
    println("Avg min loss = ",Statistics.mean(opt_losses))
    println("Std min loss = ",Statistics.std(opt_losses,corrected = false))
end