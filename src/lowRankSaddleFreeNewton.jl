using Zygote
using ElasticArrays
using LinearAlgebra

struct SFNLogger
    alpha::Float64
    rank::Float64
    losses::AbstractArray{Float64}
    spectra::AbstractArray{Float64}
end


function sherman_morrison(d_k,U_k,b,gamma)
    """
    """
    inner_vec = 1.0 ./d_k + (1.0/gamma)*ones(size(d_k))
    D = 1.0 ./ inner_vec
    UDUTb = U_k*(D.*(U_k'b))
    x_sol = ( b - (1.0/gamma)*UDUTb)/gamma
    return x_sol
end

function fullSaddleFreeNewton(f,w;hessian="full",alpha=1e0,iterations = 100,verbose = true,
                                            printing_frequency = 10,logging = true)
    """
    """
    if verbose
        print("At initial guess obj = ",f(w),"\n")
    end
    gradient(w) = Zygote.gradient(w->f(w),w)[1]
    if hessian == "full"
        FullHessian(w) = Zygote.hessian(w->f(w),w)
    end
    if logging
        losses = zeros(0)
        push!(losses,f(w))
        d = size(w)[1]
        spectra = ElasticArrays.ElasticArray{Float64}(undef, d, 0)
        logger = SFNLogger(alpha,d,losses,spectra)
    end
        
    for i = 1:iterations
        H_eigs = eigen(FullHessian(w),sortby = x -> -abs(x))
        d = H_eigs.values
        println("Largest eigenvalue is ",d[1])
        U = H_eig.vectors
        dw = U*(diagm(1 ./abs.(d))*(U'gradient(w)))
        w = w - alpha*dw
        if (i % printing_frequency == 0) & verbose
            println("At iteration ",i," obj = ",f(w),"\n")
        end
        if logging
            push!(logger.losses,f(w))
            append!(logger.spectra,H_eig.values)
        end
    end
    if logging
        return w,logger
    else
        return w,nothing
    end
end

function lowRankSaddleFreeNewton(f,w;hessian="matrix_free",alpha=1e0,gamma = 1e-5,rank = 20,iterations = 100,
                                            verbose = true,printing_frequency = 10,logging=true,log_full_spectrum = false)
    """
    """
    @assert rank <= size(w)[1]
    if verbose
        println("At initial guess obj = ",f(w),"\n")
    end
    gradient(w) = Zygote.gradient(w->f(w),w)[1]
    if hessian == "full"
        FullHessian(w) = Zygote.hessian(w->f(w),w)
    elseif hessian == "matrix_free"
        # For now I am pre-allocating the matrix to be producted with
        # for the lambda function, but other arrays can be passed in later
        matrix = zeros(size(w)[1],1)
        HessianMatrixProduct(w,matrix) = genericHessianMatrixProduct(f,w,matrix)
    elseif hessian == "from_gradient"
        throw(DomainError(hessian,"From gradient not currently working"))
        # For now I am pre-allocating the matrix to be producted with
        # for the lambda function, but other arrays can be passed in later
        matrix = zeros(size(w)[1],1)
        try
            HessianMatrixProduct(w,matrix) = fastInaccurateHessianFromGradient(gradient,w,matrix)
            println("No issue, HessianMatrixProduct created")
        catch
            println("Issue creating Hessian from gradient, using generic mat-vec")
            HessianMatrixProduct(w,matrix) = genericHessianMatrixProduct(f,w,matrix)
        end
    else
        throw(DomainError(hessian,"invalid choice for hessian"))
    end
    if logging
        losses = zeros(0)
        push!(losses,f(w))
        spectra = ElasticArrays.ElasticArray{Float64}(undef, rank, 0)
        logger = SFNLogger(alpha,rank,losses,spectra)
    end
    for i = 1:iterations
        if hessian == "full"
            H_eigs = eigen(FullHessian(w),sortby = x -> -abs(x))
            d_reduced = H_eigs.values[begin:rank]
            U_reduced = H_eigs.vectors[:,begin:rank]
        elseif hessian == "matrix_free" || hessian == "from_gradient"
            Hmp(matrix) = HessianMatrixProduct(w,matrix)
            d_reduced,U_reduced = randomizedEVD(Hmp,size(w)[1],rank)
        end
        g = gradient(w)
        dw = sherman_morrison(abs.(d_reduced),U_reduced,gradient(w),gamma)
        w = w - alpha*dw
        if (i % printing_frequency == 0) & verbose
            print("At iteration ",i," obj = ",f(w))
            println(" lambda_1 = ",d_reduced[1]," lambda_r = ",d_reduced[end])
        end
        if logging
            push!(logger.losses,f(w))
            if log_full_spectrum
                if hessian == "full"
                    H_eigs = eigen(FullHessian(w),sortby = x -> -abs(x))
                    append!(logger.spectra,H_eigs.values)
                else
                    d_full,_ = randomizedEVD(Hmp,size(w)[1],size(w)[1])
                    append!(logger.spectra,d_full)
                end
            else
                append!(logger.spectra,d_reduced)
            end
        end
    end
    if logging
        return w,logger
    else
        return w,nothing
    end
end






