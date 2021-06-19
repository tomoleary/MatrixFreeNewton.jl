using Zygote
using ElasticArrays

struct GDLogger
    alpha::Float64
    losses::AbstractArray{Float64}
end

function gradientDescent(f,w;alpha=1e0,iterations = 100,verbose = true,printing_frequency = 10,logging = true)
    if verbose
        print("At initial guess obj = ",f(w),"\n")
    end
    gradient(w) = Zygote.gradient(w->f(w),w)[1]
    if logging
        losses = zeros(0)
        push!(losses,f(w))
        logger = GDLogger(alpha,losses)
    end
    for i = 1:iterations
        dw = gradient(w)
        w = w - alpha*dw
        if (i % printing_frequency == 0) & verbose
            print("At iteration ",i," obj = ",f(w),"\n")
        end
        if logging
            push!(logger.losses,f(w))
        end
    end
    if logging
        return w,logger
    else
        return w,nothing
    end
end

struct CSGDLogger
    alphas::AbstractArray{Float64}
    losses::AbstractArray{Float64}
end

function curvatureScaledGradientDescent(f,w;iterations = 100,verbose = true,printing_frequency = 10,logging = true)
    if verbose
        print("At initial guess obj = ",f(w),"\n")
    end
    gradient(w) = Zygote.gradient(w->f(w),w)[1]
    Hessian(w) = Zygote.hessian(w->f(w),w)
    if logging
        alphas = zeros(0)
        push!(alphas,0)
        losses = zeros(0)
        push!(losses,f(w))
        logger = CSGDLogger(alphas,losses)
    end
    for i = 1:iterations
        dw = gradient(w)
        H_eig = eigen(Hessian(w),sortby = x -> -abs(x))
        alpha = 1.0/abs(H_eig.values[1])
        w = w - alpha*dw
        if (i % printing_frequency == 0) & verbose
            print("At iteration ",i," obj = ",f(w),"\n")
        end
        if logging
            push!(logger.losses,f(w))
        end
    end
    if logging
        return w,logger
    else
        return w,nothing
    end
end