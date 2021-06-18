
module ScalableNewton
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

