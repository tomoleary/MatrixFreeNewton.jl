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

module MatrixFreeNewton
	using Zygote
	using ElasticArrays
	using SparseDiffTools
	using LinearAlgebra
	using Random
	export fullNewton, gradientDescent, curvatureScaledGradientDescent, fullSaddleFreeNewton, lowRankSaddleFreeNewton
	include("fullNewton.jl")
	include("gradientDescent.jl")
	include("hessians.jl")
	include("lowRankSaddleFreeNewton.jl")
	include("randomizedEVD.jl")
end

