import numpy as np
import time
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


# Function to construct the D matrix
def Matr_G(matrix):
    # Handle division by zero or invalid entries
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute the reciprocal of the column-wise sum of the matrix
        G = np.divide(1., np.asarray(matrix.sum(axis=0)).reshape(-1))
    # Replace NaN entries with zero
    np.ma.masked_array(G, ~np.isfinite(G)).filled(0)
    # Return the sparse diagonal matrix constructed from the vector G
    return sp.diags(G)


# Function to perform Power Method while storing intermediate matrices
def PowerMethod_Storing(G, m=0.15, Tol=1e-05):
    # Get the number of nodes in the graph
    n = G.shape[0]
    # Track the time taken for the algorithm
    start = time.time()
    # Initialize a vector of ones
    e = np.ones((n, 1))
    # List to store the teleportation probabilities
    Zj = []

    # Calculate teleportation probability for each node
    for i in range(n):
        # Count non-zero elements in the i-th column
        cont = G[:, i].count_nonzero()
        if cont > 0:
            # If non-zero entries exist, assign m/n as the teleportation probability
            Zj.append(m / n)
        else:
            # For columns with only zeros, assign a uniform probability 1/n
            Zj.append(1. / n)

    # Convert the list into a NumPy array for efficient operations
    Zj_arr = np.asarray(Zj)
    # Initialize the PageRank vector with ones
    Xk = np.ones((n, 1))
    # Previous iteration's PageRank vector, initialized to 1/n
    Xk_1 = np.ones((n, 1)) / n

    # Iterate until the difference between current and previous vectors is less than tolerance
    while np.linalg.norm(Xk_1 - Xk, np.inf) > Tol:
        Xk = Xk_1
        # Compute the teleportation term
        Zxk = Zj_arr.dot(Xk)
        # Update the PageRank vector using the power iteration formula
        Xk_1 = (1 - m) * G.dot(Xk) + e * Zxk

    # Report the time taken for this method
    print(f"Time taken using Power Method with Matrix Storage (Tol={Tol}, Damping={m}): ", time.time() - start, "seconds")
    # Normalize the final PageRank vector and return it
    Xk_1 = Xk_1 / np.sum(Xk_1)
    return Xk_1


#######################################################################


# Function to create an L matrix (with non-zero entries indexed by column)
def Matr_L(nrows, ncols):
    # Initialize a dictionary to store non-zero indices by column
    ind = {}
    # Iterate over non-zero entries and store row indices for each column
    for i in range(len(ncols)):
        ind_col = ncols[i]
        # If the column is already in the dictionary, append the row index
        if ind_col in ind:
            ind[ind_col] = np.append(ind[ind_col], nrows[i])
        else:
            # Otherwise, initialize a new list with the current row index
            ind[ind_col] = np.asarray([nrows[i]])
    # Return the dictionary with column indices as keys and row indices as values
    return ind


#######################################################################


# Function to perform Power Method without storing matrices
def PowerMethod_NotStoring(matrix, m, Tol):
    # Get the number of nodes in the graph
    n = matrix.shape[0]
    # Track the execution time for this method
    start = time.time()
    # Create a dictionary for the non-zero elements' indices
    ind = Matr_L(matrix.nonzero()[0], matrix.nonzero()[1])
    # Initialize the PageRank vector with uniform values
    x = np.ones((n, 1)) / n
    # The previous iteration's vector, initialized to ones
    xc = np.ones((n, 1))

    # Iterate until convergence (when the change between iterations is smaller than tolerance)
    while np.linalg.norm(x - xc, np.inf) > Tol:
        xc = x
        # Initialize a zero vector for the new iteration
        x = np.zeros((n, 1))

        # Update the new iteration's values using the non-stored matrix multiplication
        for j in range(n):
            if j in ind:
                # For columns with non-zero entries, distribute the probability
                if len(ind[j]) != 0:
                    x[ind[j]] = x[ind[j]] + xc[j] / len(ind[j])
                else:
                    x = x + xc[j] / n
            else:
                x += xc[j] / n

        # Apply the power iteration update
        x = (1 - m) * x + m / n

    # Output the execution time for this method
    print(f"Time taken using Power Method without Matrix Storage (Tol={Tol}, Damping={m}): ", time.time() - start, "seconds")
    # Normalize the final PageRank vector and return it
    return x / np.sum(x)


#######################################################################


# Read the data and initialize matrices
matr = sio.mmread("p2p-Gnutella30.mtx")
Matr = sp.csr_matrix(matr)
D = Matr_G(Matr)
A = sp.csr_matrix(Matr.dot(D))

# Set predefined damping factors and tolerance values
DampingFactors = [0.15, 0.25, 0.50]
Tols = [1e-05, 1e-10]

# Run both methods with the predefined damping factors and different tolerance values
for DampFact in DampingFactors:
    for Tol in Tols:
        print(f"\nRunning Power Method with Matrix Storage (Damping={DampFact}, Tolerance={Tol}):")
        PR_Storing = PowerMethod_Storing(A, DampFact, Tol)
        print("Normalized PR Vector (with matrix storage):\n", np.round(PR_Storing, 6))

        print(f"\nRunning Power Method without Matrix Storage (Damping={DampFact}, Tolerance={Tol}):")
        PR_NotStoring = PowerMethod_NotStoring(A, DampFact, Tol)
        print("Normalized PR Vector (without matrix storage):\n", np.round(PR_NotStoring, 6))
