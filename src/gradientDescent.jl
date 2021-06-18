using Zygote
using ElasticArrays

function gradientDescent(f,w;alpha=1e0,iterations = 100,verbose = true,printing_frequency = 10,history = false)
    if verbose
        print("At initial guess obj = ",f(w),"\n")
    end
    gradient(w) = Zygote.gradient(w->f(w),w)[1]
    if history
        trace = ElasticArrays.ElasticArray{Float64}(undef, size(w)[1], 0)
        append!(trace,copy(w))
    end
    for i = 1:iterations
        dw = gradient(w)
        w = w - alpha*dw
        if (i % printing_frequency == 0) & verbose
            print("At iteration ",i," obj = ",f(w),"\n")
        end
        if history
            append!(trace,copy(w))
        end
    end
    if history
        return w,trace
    else
        return w,nothing
    end
end



function curvatureScaledGradientDescent(f,w;iterations = 100,verbose = true,printing_frequency = 10,history = false)
    if verbose
        print("At initial guess obj = ",f(w),"\n")
    end
    gradient(w) = Zygote.gradient(w->f(w),w)[1]
    Hessian(w) = Zygote.hessian(w->f(w),w)
    if history
        trace = ElasticArrays.ElasticArray{Float64}(undef, size(w)[1], 0)
        append!(trace,copy(w))
    end
    for i = 1:iterations
        dw = gradient(w)
        H_eig = eigen(Hessian(w),sortby = x -> -abs(x))
        alpha = 1.0/abs(H_eig.values[1])
        w = w - alpha*dw
        if (i % printing_frequency == 0) & verbose
            print("At iteration ",i," obj = ",f(w),"\n")
        end
        if history
            append!(trace,copy(w))
        end
    end
    if history
        return w,trace
    else
        return w,nothing
    end
end