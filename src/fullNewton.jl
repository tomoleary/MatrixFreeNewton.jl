using Zygote
using ElasticArrays

struct NewtonLogger
    alpha::Float64
    losses::AbstractArray{Float64}
end

function fullNewton(f,w;alpha=1e0,iterations = 2,printing_frequency = 1,logging = true)
    print("At initial guess obj = ",f(w),"\n")
    g_function(w) = Zygote.gradient(w->f(w),w)[1]
    H_function(w) = Zygote.hessian(w->f(w),w)
    if logging
        losses = zeros(0)
        push!(losses,f(w))
        logger = NewtonLogger(alpha,losses)
    end
    for i = 1:iterations
        dw = H_function(w)\g_function(w)
        w = w - alpha*dw
        if i % printing_frequency == 0
            print("At iteration ",i," obj = ",f(w),"\n")
#             print("w = ",w,"\n")
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

