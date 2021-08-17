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

struct GDLogger
    alpha::Float64
    losses::AbstractArray{Float64}
    spectra::AbstractArray{Float64}
end

function gradientDescent(f,w;alpha=1e0,iterations = 100,verbose = true,printing_frequency = 10,
                                                            logging = true,log_full_spectrum = false)
    if verbose
        print("At initial guess obj = ",f(w),"\n")
    end
    gradient(w) = Zygote.gradient(w->f(w),w)[1]
    if logging
        losses = zeros(0)
        push!(losses,f(w))
        d = size(w)[1]
        spectra = ElasticArrays.ElasticArray{Float64}(undef, d, 0)
        logger = GDLogger(alpha,losses,spectra)
        if log_full_spectrum
            Hessian(w) = Zygote.hessian(w->f(w),w)
        end
    end
    for i = 1:iterations
        dw = gradient(w)
        w = w - alpha*dw
        if (i % printing_frequency == 0) & verbose
            print("At iteration ",i," obj = ",f(w),"\n")
        end
        if logging
            push!(logger.losses,f(w))
            if log_full_spectrum
                H_eig = eigen(Hessian(w),sortby = x -> -abs(x))
                append!(logger.spectra,H_eig.values)
            end
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
    spectra::AbstractArray{Float64}
end

function curvatureScaledGradientDescent(f,w;iterations = 100,verbose = true,printing_frequency = 10,
                                                                logging = true,log_full_spectrum = false)
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
        d = size(w)[1]
        spectra = ElasticArrays.ElasticArray{Float64}(undef, d, 0)
        logger = CSGDLogger(alphas,losses,spectra)
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
            if log_full_spectrum
                append!(logger.spectra,H_eig.values)
            end
        end
    end
    if logging
        return w,logger
    else
        return w,nothing
    end
end