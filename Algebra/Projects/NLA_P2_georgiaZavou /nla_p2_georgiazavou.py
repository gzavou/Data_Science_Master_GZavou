
import pandas as pd
import numpy as np
from numpy import genfromtxt, vstack, sqrt, std, concatenate, reshape, dot
from numpy.linalg import norm, svd
from numpy.core.fromnumeric import argmin
import sys
from scipy.linalg import solve_triangular, qr
import matplotlib.pyplot as plt
import imageio
from imageio import imread, imsave
from pandas import read_csv, DataFrame, concat

# SVD FOR LS
def svd_LS(A, b):
    U, Sigma, VT = np.linalg.svd(A, full_matrices=False)

    Sigma_inv = np.diag(1 / Sigma)

    x_svd = VT.T @ Sigma_inv @ U.T @ b

    return x_svd

import numpy as np
from scipy.linalg import qr, solve_triangular

# QR factorization for solving the Least Squares problem
# This function handles both full-rank and rank-deficient matrices.
def qr_LS(A, b):
    # Step 1: Compute the rank of matrix A
    # Rank gives us the number of linearly independent columns of A
    Rank = np.linalg.matrix_rank(A)

    # Initialize the solution vector
    x_qr = None

    # Step 2: Case for Full-rank matrix (Rank == number of columns of A)
    if Rank == A.shape[1]:
        # Substep 2.1: Perform QR factorization on A (A = Q * R)
        # Q_fullr is an orthogonal matrix, R_fullr is an upper triangular matrix
        Q_fullr, R_fullr = np.linalg.qr(A)

        # Substep 2.2: Solve the system using back substitution
        # First, calculate the intermediate vector y_aux = Q^T * b
        y_aux = np.transpose(Q_fullr).dot(b)

        # Then, solve R * x = y_aux using back substitution
        x_qr = solve_triangular(R_fullr, y_aux)

    # Step 3: Case for Rank-deficient matrix (when Rank < number of columns of A)
    else:
        # Substep 3.1: QR factorization with pivoting for rank-deficient matrix
        # The pivoting step helps handle situations where the matrix is poorly conditioned
        Q, R, P = qr(A, mode='economic', pivoting=True)

        # Substep 3.2: Extract the relevant upper-left submatrix of R for rank-deficient case
        # We use the leading 'Rank' rows and columns of R, which correspond to the rank of A
        R_def = R[:Rank, :Rank]

        # Substep 3.3: Compute the vector c = Q^T * b, but only use the first 'Rank' components
        # This reduces the system to the rank-deficient case, where R is of size (Rank x Rank)
        c = np.transpose(Q).dot(b)[:Rank]

        # Substep 3.4: Solve the system R_def * u = c for u using back substitution
        u = solve_triangular(R_def, c)

        # The remaining components correspond to the "deficiency" in the rank, so set them to zero
        # These components are the parts of the solution that correspond to the null space
        v = np.zeros((A.shape[1] - Rank))

        # Substep 3.5: Combine the solutions for the full least squares solution
        # Concatenate the solution vectors u (corresponding to the rank part) and v (zeroed out)
        # The permutation matrix P is used to adjust the solution to the correct order
        x_qr = np.linalg.solve(np.transpose(np.eye(A.shape[1])[:, P]), np.concatenate((u, v)))

    # Step 4: Return the solution vector x_qr
    return x_qr



# Load datasets
def datafile(degree):
    data = genfromtxt("dades.csv", delimiter="   ")
    points, b = data[:, 0], data[:, 1]

    A = vstack([points ** d for d in range(degree)]).T

    return A, b

def datafile2(degree):
    data = genfromtxt('dades_regressio.csv', delimiter=',')
    A, b = data[:, :-1], data[:, -1]
    return A, b

svd_errors = []
degrees=range(3,10)

# Call functions with datafile
for degree in range(3,10):
    A, b = datafile(degree)
    x_svd = svd_LS(A, b)
    x_qr = qr_LS(A, b)
    svd_errors.append(norm(A.dot(x_svd) - b))

min_svd_error_pos = argmin(svd_errors)
best_degree = min_svd_error_pos+3

# Compare results for datafile
print("First Datafile:")
print("Best degree:", best_degree)
A, b = datafile(best_degree)
x_svd = svd_LS(A, b)
x_qr = qr_LS(A, b)
print("\n")
print("LS solution with SVD:", x_svd)
print("\nNorm of solution:", norm(x_svd))
print("\nThe Error is :", norm(A.dot(x_svd)-b))
print("\n\n")
print("\nLS solution with QR:", x_qr)
print("\nNorm of solution: ", norm(x_qr))
print("\nThe Error is :", norm(A.dot(x_qr)-b))
print("\n")

print("Datafile2:")
print("Best degree:", best_degree)
A, b = datafile2(best_degree)
x_svd = svd_LS(A, b)
x_qr = qr_LS(A, b)
print("\n")
print("LS solution using SVD:", x_svd)
print("LS  Norm:", norm(x_svd))
print("\nThe Error is :", norm(A.dot(x_svd)-b))
print("\n\n")
print("LS  with QR:", x_qr)
print("\nNorm:", norm(x_qr))
print("\nThe Error is :", norm(A.dot(x_qr)-b))
print("\n")

image1 = imageio.imread('butterfly.jpg')

# And an image with text is going t
image2 = imageio.imread('letters.jpg')

Image1 = image1[:,:,1]
Image2 = image2[:,:,1]

def compress_image(matrix, output_prefix='compressed'):
    U, sigma, V = np.linalg.svd(matrix)
    rank= [1, 5, 25, 50, 100]

    for i in rank:
        A = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])

        relative_error = np.sum(sigma[i:]**2) / np.sum(sigma**2)

        # Create new files with the name and Frobenius norm c
        percentage_captured = np.linalg.norm(A) / np.linalg.norm(matrix)

        # Save the compressed image
        output_filename = f"{output_prefix}_rank_{i}_capture_{percentage_captured:.2f}.jpg"
        imageio.imwrite(output_filename, np.clip(A, 0, 255).astype(np.uint8))

        print(f"Rank {i} - Percentage Captured: {percentage_captured:.2f}% - Relative Error: {relative_error:.4f}")

print("Image 1 with the butterfly :")
compress_image(Image1, output_prefix='butterfly_compressed')
print("\n")
print("Image 2 with letters Results:")
compress_image(Image2, output_prefix='letters_compressed')

#  Principal Component Analysis

# Function that read data from exaple.dat
def read_txt():
    X = np.genfromtxt('example.dat', delimiter = ' ')
    return X.T

# Function that read data from the csv file
def read_csv():
    X = np.genfromtxt('RCsGoff.csv', delimiter = ',')
    # Get rid of unnecessary variables
    return X[1:,1:].T

import numpy as np
from scipy.linalg import qr, solve_triangular

# QR factorization for solving the Least Squares problem
# This function handles both full-rank and rank-deficient matrices.
def qr_LS(A, b):
    # Step 1: Compute the rank of matrix A
    # Rank gives us the number of linearly independent columns of A
    Rank = np.linalg.matrix_rank(A)

    # Initialize the solution vector
    x_qr = None

    # Step 2: Case for Full-rank matrix (Rank == number of columns of A)
    if Rank == A.shape[1]:
        # Substep 2.1: Perform QR factorization on A (A = Q * R)
        # Q_fullr is an orthogonal matrix, R_fullr is an upper triangular matrix
        Q_fullr, R_fullr = np.linalg.qr(A)

        # Substep 2.2: Solve the system using back substitution
        # First, calculate the intermediate vector y_aux = Q^T * b
        y_aux = np.transpose(Q_fullr).dot(b)

        # Then, solve R * x = y_aux using back substitution
        x_qr = solve_triangular(R_fullr, y_aux)

    # Step 3: Case for Rank-deficient matrix (when Rank < number of columns of A)
    else:
        # Substep 3.1: QR factorization with pivoting for rank-deficient matrix
        # The pivoting step helps handle situations where the matrix is poorly conditioned
        Q, R, P = qr(A, mode='economic', pivoting=True)

        # Substep 3.2: Extract the relevant upper-left submatrix of R for rank-deficient case
        # We use the leading 'Rank' rows and columns of R, which correspond to the rank of A
        R_def = R[:Rank, :Rank]

        # Substep 3.3: Compute the vector c = Q^T * b, but only use the first 'Rank' components
        # This reduces the system to the rank-deficient case, where R is of size (Rank x Rank)
        c = np.transpose(Q).dot(b)[:Rank]

        # Substep 3.4: Solve the system R_def * u = c for u using back substitution
        u = solve_triangular(R_def, c)

        # The remaining components correspond to the "deficiency" in the rank, so set them to zero
        # These components are the parts of the solution that correspond to the null space
        v = np.zeros((A.shape[1] - Rank))

        # Substep 3.5: Combine the solutions for the full least squares solution
        # Concatenate the solution vectors u (corresponding to the rank part) and v (zeroed out)
        # The permutation matrix P is used to adjust the solution to the correct order
        x_qr = np.linalg.solve(np.transpose(np.eye(A.shape[1])[:, P]), np.concatenate((u, v)))

    # Step 4: Return the solution vector x_qr
    return x_qr

# We will creat a function that apply PCA analysis
def PCA(matrix_choice, file_choice):

    # Choose the data
    if file_choice == 1:# text dataset
        X = read_txt()
    else:# csv dataset
        X = read_csv()

    # Substract the mean
    X = X - np.mean(X, axis = 0)
    n = X.shape[0]
    # Choose the matrix and complete the program
    if matrix_choice == 1:# covariance matrix
        Y = (1 / (np.sqrt( n - 1))) * X.T
        U,S,VH = np.linalg.svd(Y, full_matrices = False)

        # Portion of the total variance accumulated in each of the PC
        total_var = S**2 / np.sum(S**2)
        # Standard deviation of each of the PC
        # Observe that the matrix V contains the eigenvectors of Cx
        standard_dev = np.std(VH, axis = 0)

 # Expression of the original dataset in the new PCA coordinates
        new_expr_PCA_coord = np.matmul(VH,X).T
    else:# correlation matrix
        X = (X.T / np.std(X, axis = 1)).T
        Y = (1 / (np.sqrt( n - 1))) * X.T
        U,S,VH = np.linalg.svd(Y, full_matrices = False)

        # Portion of the total variance accumulated in each of the PC
        total_var = S**2 / np.sum(S**2)

        # Standard deviation of each of the PC
        # Observe that the matrix V contains the eigenvectors of Cx
        standard_dev = np.std(VH.T, axis = 0)

        # Expression of the original dataset in the new PCA coordinates
        new_expr_PCA_coord = np.matmul(VH,X).T
    return total_var, standard_dev, new_expr_PCA_coord, S

def Scree_plot(S,number_figure,matrix_type):
    if matrix_type == 1:#covariance matrix
        plt.figure(number_figure)
        plt.plot(range(len(S)), S)
        for i in range(len(S)):
            plt.scatter(i,S[i],color='purple')
        plt.title('Scree plot for the covariance matrix')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.savefig("scree_plot_cov.jpg")
        plt.show()
    else:#correlation matrix
        plt.figure(number_figure)
        plt.plot(range(len(S)), S)
        for i in range(len(S)):
            plt.scatter(i,S[i],color='purple')
        plt.title('Scree plot for the correlation matrix')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.savefig("scree_plot_corr.jpg")
        plt.show()

def Kasier(S):
    count = 0
    for i in range(len(S)):
        if S[i]>1:
            count += 1
    return count

def rule_34(var):
    total_var = sum(var)
    new_var = []
    i = 0

    while sum(new_var) < 3*total_var/4:
        new_var.append(var[i])
        i += 1

    return len(new_var)

# Covariance matrix
print('Covariance matrix')
total_var,standar_dev,new_expr,S = PCA(1,1)
print('\n')
print('Accumulated total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,1,1)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var))
print('\n')

# Correlation matrix
print('Correlation matrix')
total_var,standar_dev,new_expr,S = PCA(0,1)
print('\n')
print('Accumulated total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,2,0)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var))
print('\n')

# Covariance matrix
print('Covariance matrix')
total_var,standar_dev,new_expr,S = PCA(1,0)
print(new_expr.shape)
print('\n')
print('Accumulated total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,3,1)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var))
print('\n')

# Save results to a file
# Call the read_csv function
X_RCsGoff = read_csv()

# Convert NumPy arrays to DataFrame
data_df = pd.DataFrame(data=new_expr[:20, :].T, columns=[f"PC{i}" for i in range(1, 21)])
variance_df = pd.DataFrame(data=reshape(total_var, (20, 1)), columns=["Variance"])

# Assuming 'gene' column exists, drop it
if 'gene' in data_df.columns:
    data_df = data_df.drop('gene', axis=1)

# Assuming 'Sample' is the index
data_df.index.name = "Sample"

# Add the variance column to the DataFrame
data_df["Variance"] = variance_df["Variance"]

# Save to a text file
data_df.to_csv("rcsgoff_covariance.txt", sep='\t')

# Correlation matrix
print('Correlation matrix')
total_var,standar_dev,new_expr,S = PCA(0,0)
print(new_expr.shape)
print('\n')
print('Accumulated total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,4,0)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var))
print('\n')

# Save results to a file
# Call the read_csv function
X_RCsGoff = read_csv()

# Convert NumPy arrays to DataFrame
data_df = pd.DataFrame(data=new_expr[:20, :].T, columns=[f"PC{i}" for i in range(1, 21)])
variance_df = pd.DataFrame(data=reshape(total_var, (20, 1)), columns=["Variance"])

# Assuming 'gene' column exists, drop it
if 'gene' in data_df.columns:
    data_df = data_df.drop('gene', axis=1)

# Assuming 'Sample' is the index
data_df.index.name = "Sample"

# Add the variance column to the DataFrame
data_df["Variance"] = variance_df["Variance"]

# Save to a text file
data_df.to_csv("rcsgoff_correlation.txt", sep='\t')