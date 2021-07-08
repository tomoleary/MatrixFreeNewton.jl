using Pkg; Pkg.activate(".")
using Random, NPZ, Statistics
using MatrixFreeNewton


function rosenbrockn(x)
    f = 0.0
    for i = 1:size(x)[1]-1
        f += 100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2
    end
    return f
end

n = 10

iterations = 1000
lrsfn_rank = Int(floor(0.9*n))

# Allocate logger dictionaries for seeding
loggers_gd = Dict()
loggers_csgd = Dict()
loggers_newton = Dict()
loggers_sfn = Dict()
loggers_lrsfn = Dict()
seed = 0


x_0 = zeros(n)

println("Now for gradient descent ")
w_star_gd,logger_gd = gradientDescent(rosenbrockn,x_0,alpha = 1e-3,iterations = iterations)
loggers_gd[seed] = logger_gd

println("Now for curvature gradient descent ")
w_star_gd,logger_csgd = curvatureScaledGradientDescent(rosenbrockn,x_0,iterations = iterations)
loggers_csgd[seed] = logger_csgd

print("Now for Newton \n")
w_star_newton,logger_newton = fullNewton(rosenbrockn,x_0,alpha = 1e0,iterations=15)
loggers_newton[seed] = logger_newton

println("Now for low rank SFNewton with full rank Hessian ")
w_star_newton,logger_sfn = lowRankSaddleFreeNewton(rosenbrockn,x_0,printing_frequency=iterations,
                                                                        iterations = 15,rank = n)
loggers_sfn[seed] = logger_sfn

println("Now for low rank SFNewton with reduced Hessian with LRSFN rank = ",lrsfn_rank)
w_star_newton,logger_lrsfn = lowRankSaddleFreeNewton(rosenbrockn,x_0,alpha = 1e-2,printing_frequency=10,
                                                    gamma = 1e0,iterations = iterations,rank = lrsfn_rank,
                                                    log_full_spectrum = true)
loggers_lrsfn[seed] = logger_lrsfn


println("Done")

# Save data for post-processing
data_dir = "rosenbrockn_data/"
if ~isdir(data_dir)
    mkdir(data_dir)
end
problem_name = "rosenbrock_d="*string(n)
optimizers = ["gd","csgd","newton","sfn","lrsfn"]
logger_dicts = [loggers_gd,loggers_csgd,loggers_newton,loggers_sfn,loggers_lrsfn]

for (optimizer,logger_dict) in zip(optimizers,logger_dicts)
    println("optimizer = ",optimizer)
    opt_losses = zeros(0)
    for (seed,logger) in logger_dict
        name = problem_name*optimizer*"_"*string(seed)
        if optimizer == "lrsfn"
            name *="rank_"*string(logger.rank)
        end
        println("name = ",name)
        # Save losses
        npzwrite(data_dir*name*"_losses.npy",logger.losses)
        min_loss = minimum(logger.losses)
        append!(opt_losses,min_loss)
        # If sfn save spectrum:
        if optimizer in ["sfn","lrsfn"]
            
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



