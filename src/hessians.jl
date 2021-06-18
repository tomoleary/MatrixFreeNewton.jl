using Zygote
using SparseDiffTools


function fullHessian(f,w)
	fullH(w) = Zygote.hessian(w->f(w),w)
	return fullH(w)
end

function genericHessianMatrixProduct(f,w,v)
    g(w) = Zygote.gradient(w->f(w),w)[1]
    Hmp(w,v) = Zygote.forward_jacobian(w->g(w)'v,w)[2]
    return Hmp(w,v)
end

function fastInaccurateHessianFromGradient(g,w,v)
    sHgv = SparseDiffTools.HesVecGrad(g,w)
    Hv = zero(v)
    for i=1:size(v)[2]
        Hv[:,i] = sHgv*v[:,i]
    end
    return Hv
end

