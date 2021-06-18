using Zygote
using ElasticArrays

function fullNewton(f,w;alpha=1e0,iterations = 2,printing_frequency = 1,history = false)
    print("At initial guess obj = ",f(w),"\n")
    g_function(w) = Zygote.gradient(w->f(w),w)[1]
    H_function(w) = Zygote.hessian(w->f(w),w)
    if history
        trace = ElasticArrays.ElasticArray{Float64}(undef, size(w)[1], 0)
        append!(trace,copy(w))
    end
    for i = 1:iterations
        dw = H_function(w)\g_function(w)
        w = w - alpha*dw
        if i % printing_frequency == 0
            print("At iteration ",i," obj = ",f(w),"\n")
#             print("w = ",w,"\n")
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