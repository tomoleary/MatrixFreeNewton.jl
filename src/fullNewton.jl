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
using ElasticArrays

struct NewtonLogger
    alpha::Float64
    losses::AbstractArray{Float64}
end

function fullNewton(f,w;alpha=1e0,iterations = 2,printing_frequency = 10,logging = true)
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

