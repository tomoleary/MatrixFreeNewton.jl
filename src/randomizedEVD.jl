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

using LinearAlgebra
using Random

function randomizedEVD(Aop,n,rank;p=nothing, check=false,seed = 0)
    """
    """
    random_state = Random.MersenneTwister(seed)
    if p == nothing
        p =  ceil(Int, rank*0.05)
    end
    Omega = randn(random_state,n,rank)
    Y = Aop(Omega)
    Q = qr(Y).Q[:,begin:rank]
    T = LinearAlgebra.Symmetric(Q'Aop(Q))
    T_eig = eigen(T,sortby = x -> -abs(x))
    d_reduced = T_eig.values
    U_reduced = Q*T_eig.vectors
    if check
        error = Aop(U_reduced[:,1:rank]) - U_reduced[:,1:rank]*diagm(d_reduced[1:rank])
        print("||error||_F = ",norm(error),"\n")
        print("||error||_2 = ",opnorm(error),"\n")
    end
    return d_reduced, U_reduced
end 




function blockRangeFinder(Aop,n,epsilon,block_size;verbose = false,seed = 0)
    """
    Randomized algorithm for block range finding    

    Taken from http://people.maths.ox.ac.uk/martinsson/Pubs/2015_randQB.pdf
    
    Parameters:
    -----------
    Aop : {Callable} n x n symmetric matrix
          Hermitian matrix operator whose eigenvalues need to be estimated
          y = Aop(dw) is the action of A in the direction dw 
    n   : size of matrix A
    epsilon : relative reduction in error
            
    Returns:
    --------
    Q : range for Aop
    """

    random_state = Random.MersenneTwister(seed)
    w = randn(random_state,n,1)
    Action = Aop(w)
    initial_error = opnorm(Action)
    big_Q = nothing
    converged = false
    iteration = 0

    while ~converged
        # Sample Gaussian random matrix
        Omega = randn(random_state,n,block_size)
        Y = Aop(Omega)
        Q = qr(Y).Q[:,begin:block_size]
        # Update the basis
        if big_Q == nothing
            big_Q = Q
        else
            Q -= big_Q(big_Q'(Q))
            big_Q = cat(big_Q,Q,dims=(2,2))
            # This QR gets slow after many iterations, only last columns
            # need to be orthonormalized
            dimension = shape(big_Q)[2]
            big_Q = qr(big_Q).Q[:,begin:dimension]
        end
        # Error estimation
        Approximation_error = Action - big_Q*(big_Q'Action)
        error_ = opnorm(Approximation_error)
        converged = error_ < epsilon*initial_error
        iteration += 1
        if verbose
            println("At iteration ",iteration," error is ", error_, " converged = ",converged)
        end
        if iteration > n // block_size
            break
        end
    end
    return big_Q
end




function eigensolverFromRange(Aop,Q;verbose = false)
    """
    Randomized algorithm for Hermitian eigenvalue problems
    Returns k largest eigenvalues computed using the randomized algorithm
    
    
    Parameters:
    -----------
    Aop : {Callable} n x n
          Hermitian matrix operator whose eigenvalues need to be estimated
          y = Aop(dw) is the action of A in the direction dw 
    Q : Array n x r
          
            
    Returns:
    --------
    
    d : ndarray, (k,)           
        eigenvalues arranged in descending order
    U : ndarray, (n,k)
        eigenvectors arranged according to eigenvalues
    """
    m = size(Q)[2]
    T = LinearAlgebra.Symmetric(Q'Aop(Q))
    T_eig = eigen(T,sortby = x -> -abs(x))
    d_reduced = T_eig.values
    U_reduced = Q*T_eig.vectors

    return d_reduced, U_reduced

end




