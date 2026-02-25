import numpy as np
from scipy.optimize import minimize

def optimize_upper_triangular(n, p):
    """
    Finds the optimal upper-triangular matrix M minimizing the expectation E
    under the Frobenius norm constraint ||M||_F^2 = 1.
    """
    num_upper = n * (n + 1) // 2  # Number of upper-triangular elements

    # Objective function (operates only on upper-triangular elements)
    def objective(x):
        M = np.zeros((n, n))
        # Fill upper-triangular part with x
        rows, cols = np.triu_indices(n)
        M[rows, cols] = x / np.linalg.norm(x)
        # Compute the expectation terms
        term1 = p**2 * (1 - p)**2 * np.sum(M**4)
        row_sums = np.sum(M**2, axis=1)
        term2_row = np.sum(row_sums**2)
        col_sums = np.sum(M**2, axis=0)
        term2_col = np.sum(col_sums**2)
        term2 = p**3 * (1 - p) * (term2_row + term2_col)
        MtM = M.T @ M
        trace_term = np.trace(MtM @ MtM)
        term3 = p**4 * trace_term
        return term1 + term2 + term3

    # Frobenius norm constraint (sum of squares of upper-triangular elements = 1)
    def constraint(x):
        return np.sum(x**2) - 1

    # Initial guess: upper-triangular elements set to 1/sqrt(num_upper)
    # x0 = np.ones(num_upper) / np.sqrt(num_upper)
    x0 = np.random.rand(num_upper)
    x0 /= np.linalg.norm(x0)

    # Optimization
    cons = ({'type': 'eq', 'fun': constraint})
    result = minimize(objective, x0, method='BFGS')

    # Reconstruct the optimal upper-triangular matrix
    M_optimal = np.zeros((n, n))
    M_optimal[np.triu_indices(n)] = result.x
    return M_optimal

if __name__ == '__main__':
    # Example usage:
    n = 10
    p = 0.1
    optimal_M = optimize_upper_triangular(n, p)
    np.set_printoptions(edgeitems=30, linewidth=100000)
    print("Optimal Upper-Triangular Matrix M:")
    print(optimal_M)
    print("Frobenius norm squared:", np.sum(optimal_M**2))
