# This file is part of the MatrixFreeNewton.jl package
#
# hessianlearn is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2.1 of the License, or any later version.
#
# hessianlearn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

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

