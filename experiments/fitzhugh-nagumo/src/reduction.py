"""
Model reduction techniques: POD (Proper Orthogonal Decomposition) and DEIM 
(Discrete Empirical Interpolation Method) for the FitzHugh-Nagumo system.
"""

import numpy as np
from scipy.integrate import solve_ivp
from .pde_model import input_current, nonlinear_term_fhn


def deim(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete Empirical Interpolation Method (DEIM) for selecting interpolation points.
    
    Parameters
    ----------
    U : np.ndarray
        (n, m) matrix of basis vectors for the nonlinear term
        
    Returns
    -------
    indices : np.ndarray
        (m,) selected DEIM indices
    P : np.ndarray
        (n, m) DEIM projection matrix with ones at selected indices
    """
    n, m = U.shape
    indices = []

    # First index: maximum absolute value in first basis vector
    p1 = np.argmax(np.abs(U[:, 0]))
    indices.append(p1)

    # Greedy selection of remaining indices
    for ell in range(1, m):
        U_prev = U[:, :ell]
        P_prev_T_U_prev = U_prev[indices, :]
        rhs = U[indices, ell]

        c = np.linalg.solve(P_prev_T_U_prev, rhs)
        r = U[:, ell] - U_prev @ c

        p = np.argmax(np.abs(r))
        indices.append(p)

    # Create projection matrix
    P = np.eye(n)[:, indices]
    return np.array(indices), P


def rhs_reduced(
    t: float,
    y_tilde: np.ndarray,
    Ar: np.ndarray,
    B: np.ndarray,
    V_k: np.ndarray,
    indices: np.ndarray,
    r_const: np.ndarray,
    g: np.ndarray,
    h: float,
    n: int,
    epsilon: float
) -> np.ndarray:
    """
    Right-hand side for the reduced FHN model:
    dy_tilde/dt = Ar*y_tilde + B*(P^T G(V_k y_tilde)) + r_const + (epsilon/h)*i_0(t)*(V_k^T g)
    
    Parameters
    ----------
    t : float
        Current time
    y_tilde : np.ndarray
        Reduced solution vector (k,)
    Ar : np.ndarray
        (k, k) reduced linear operator matrix
    B : np.ndarray
        (k, m) DEIM matrix
    V_k : np.ndarray
        (2n, k) POD basis for the full state
    indices : np.ndarray
        (m,) DEIM selected indices
    r_const : np.ndarray
        (k,) reduced constant term
    g : np.ndarray
        (2n,) boundary input vector in full space
    h : float
        Grid spacing
    n : int
        Spatial dimension (half of full dimension, size of v and w individually)
    epsilon : float
        Timescale parameter
        
    Returns
    -------
    np.ndarray
        Time derivative dy_tilde/dt (k,)
    """
    
    
    # Sample G = [f(v)/epsilon; 0] directly at DEIM indices.
    G_sampled = np.zeros(len(indices))
    v_mask = indices < n
    v_indices = indices[v_mask]
    G_sampled[v_mask] = nonlinear_term_fhn(V_k[v_indices, :] @ y_tilde) / epsilon
    
    # Get boundary current input
    i_0_t = input_current(t)
    
    # Project boundary forcing term using POD basis
    g_r = V_k.T @ g
    boundary_forcing = epsilon * (i_0_t / h) * g_r
    
    # Full reduced dynamics
    return Ar @ y_tilde + B @ G_sampled + r_const + boundary_forcing


def solve_reduced_model(
    y0_reduced: np.ndarray,
    Ar: np.ndarray,
    B: np.ndarray,
    V_k: np.ndarray,
    indices: np.ndarray,
    r_const: np.ndarray,
    g: np.ndarray,
    h: float,
    n: int,
    epsilon: float,
    t0: float = 0.0,
    tf: float = 8.0,
    num_snapshots: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the reduced ODE system for the FHN model.
    
    Parameters
    ----------
    y0_reduced : np.ndarray
        Initial condition in reduced space (k,)
    Ar : np.ndarray
        Reduced linear operator matrix (k, k)
    B : np.ndarray
        DEIM matrix (k, m)
    V_k : np.ndarray
        POD basis for full state (2n, k)
    indices : np.ndarray
        DEIM selected indices (m,)
    r_const : np.ndarray
        Reduced constant term (k,)
    g : np.ndarray
        Boundary input vector (2n,)
    h : float
        Grid spacing
    n : int
        Spatial dimension (half of full)
    epsilon : float
        Timescale parameter
    t0 : float
        Initial time
    tf : float
        Final time
    num_snapshots : int
        Number of time snapshots
        
    Returns
    -------
    t : np.ndarray
        Time snapshots (num_snapshots,)
    Y_reduced : np.ndarray
        (k, num_snapshots) reduced solution snapshots
    """
    t_eval = np.linspace(t0, tf, num_snapshots)
    
    sol_reduced = solve_ivp(
        fun=lambda t, y: rhs_reduced(t, y, Ar, B, V_k, indices, r_const, g, h, n, epsilon),
        t_span=(t0, tf),
        y0=y0_reduced,
        t_eval=t_eval,
        method="BDF"
    )
    
    if not sol_reduced.success:
        raise RuntimeError(f"ODE solver failed: {sol_reduced.message}")
    
    return sol_reduced.t, sol_reduced.y


def compute_pod_basis(Y: np.ndarray, tol: float = 1e-2):
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    
    rel_diff = np.abs(np.diff(s)) / s[:-1]
    k = np.argmax(rel_diff < tol) + 1 if np.any(rel_diff < tol) else len(s)
    # energy = np.cumsum(s**2) / np.sum(s**2)
    # k = np.argmax(energy >= (1 - tol)) + 1  
    k = 15
    return U[:, :k], s 
