"""
PDE model definitions and setup functions for the 1D FitzHugh-Nagumo system.

System:
    ε v_t = ε² v_xx + f(v) - w + c
    w_t = b v - γ w + c
    
with f(v) = v(v - 0.1)(1 - v) and Neumann BCs:
    v_x(0, t) = -i_0(t)
    v_x(1, t) = 0
"""

import numpy as np
from scipy.integrate import solve_ivp


# FHN model parameters
EPSILON = 0.015
B = 0.5
GAMMA = 2.0
C = 0.05


def build_neumann_laplacian_1d(n: int) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Build the 1D Laplacian matrix with Neumann boundary conditions.
    
    For domain [0, L] = [0, 1] with N spatial nodes:
        h = L / (N - 1) = 1 / (N - 1)
        x_i = (i - 1) * h, i = 1, ..., N
    
    The Neumann BCs are handled by modifying the first and last rows:
    - Left BC: v_x(0, t) = -i_0(t) modifies first row
    - Right BC: v_x(1, t) = 0 modifies last row
    
    Parameters
    ----------
    n : int
        Number of spatial grid points (N)
        
    Returns
    -------
    D : np.ndarray
        (N, N) discrete Laplacian matrix with Neumann BCs
    h : float
        Grid spacing
    x : np.ndarray
        Spatial grid points
    """
    h = 1.0 / (n - 1)
    
    # Build the standard second-derivative matrix
    # D_ii = -2, D_{i,i±1} = 1, scaled by 1/h²
    main = -2.0 * np.ones(n)
    off = 1.0 * np.ones(n - 1)
    D = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    D = (1.0 / h**2) * D
    
    # Modify first row for left Neumann BC: v_x(0, t) = -i_0(t)
    # Using ghost point elimination:
    #   v_0 = v_1 + h * i_0(t)
    #   v_xx(x_1) ≈ (v_0 - 2*v_1 + v_2) / h² = (v_2 - v_1) / h² + i_0(t) / h
    # So first row becomes: [−1, 1, 0, ..., 0] / h²
    D[0, :] = 0.0
    D[0, 0] = -1.0 / h**2
    D[0, 1] = 1.0 / h**2
    
    # Modify last row for right Neumann BC: v_x(L, t) = 0
    # Using ghost point elimination:
    #   v_{N+1} = v_N
    #   v_xx(x_N) ≈ (v_{N-1} - 2*v_N + v_{N+1}) / h² = (v_{N-1} - v_N) / h²
    # So last row becomes: [0, ..., 0, 1, −1] / h²
    D[n - 1, :] = 0.0
    D[n - 1, n - 2] = 1.0 / h**2
    D[n - 1, n - 1] = -1.0 / h**2
    
    x = np.linspace(0.0, 1.0, n)
    return D, h, x


def boundary_input_vector(n: int) -> np.ndarray:
    """
    Build the boundary input vector g for the left Neumann BC.
    
    Only the first equation sees the i_0(t) forcing, so:
        g = [1, 0, ..., 0]ᵀ ∈ ℝᴺ
    
    Parameters
    ----------
    n : int
        Number of spatial grid points
        
    Returns
    -------
    g : np.ndarray
        Boundary input vector
    """
    g = np.zeros(n)
    g[0] = 1.0
    return g


def input_current(t: float) -> float:
    """
    Boundary current input: i_0(t) = 5·10⁴ · t³ · exp(-15t)
    
    Parameters
    ----------
    t : float
        Time
        
    Returns
    -------
    float
        Current value at time t
    """
    return 5.0e4 * t**3 * np.exp(-15.0 * t)


def nonlinear_term_fhn(v: np.ndarray) -> np.ndarray:
    """
    FitzHugh-Nagumo nonlinearity: f(v) = v(v - 0.1)(1 - v)
    
    Parameters
    ----------
    v : np.ndarray
        Voltage vector (N,)
        
    Returns
    -------
    np.ndarray
        Nonlinear term f(v)
    """
    return v * (v - 0.1) * (1.0 - v)


def rhs(
    t: float,
    y: np.ndarray,
    D: np.ndarray,
    g: np.ndarray,
    h: float,
    n: int,
    epsilon: float,
    b: float,
    gamma: float,
    c: float
) -> np.ndarray:
    """
    Right-hand side of the semidiscrete FHN system.
    
    The full system in block form is:
        d/dt [v]   = [ε D              -1/ε I  ] [v]   + [1/ε · f(v)    ]
             [w]     [b I              -γ I    ] [w]     [0              ]
                     + [c/ε · 1  ]   + [ε/h · i_0(t) · g]
                       [c · 1    ]     [0               ]
    
    Parameters
    ----------
    t : float
        Current time
    y : np.ndarray
        State vector [v; w] ∈ ℝ²ᴺ
    D : np.ndarray
        Discrete Laplacian (N, N)
    g : np.ndarray
        Boundary input vector (N,)
    h : float
        Grid spacing
    n : int
        Number of spatial nodes (dimension of v and w individually)
    epsilon : float
        Timescale separation parameter ε
    b : float
        Parameter b in w equation
    gamma : float
        Parameter γ in w equation
    c : float
        Constant c
        
    Returns
    -------
    dydt : np.ndarray
        Time derivative [dv/dt; dw/dt] ∈ ℝ²ᴺ
    """
    v = y[:n]
    w = y[n:]
    
    # Get boundary current input
    i_0_t = input_current(t)
    
    # Compute v equation: v̇ = ε D v + (1/ε) f(v) - (1/ε) w + (c/ε) 1 + ε (i_0(t)/h) g
    dv_dt = (
        epsilon * (D @ v)
        + (1.0 / epsilon) * nonlinear_term_fhn(v)
        - (1.0 / epsilon) * w
        + (c / epsilon) * np.ones(n)
        + epsilon * (i_0_t / h) * g
    )
    
    # Compute w equation: ẇ = b v - γ w + c 1
    dw_dt = b * v - gamma * w + c * np.ones(n)
    
    # Stack into single vector
    dydt = np.concatenate([dv_dt, dw_dt])
    
    return dydt


def initial_condition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Initial conditions for the FHN system: v(x, 0) = 0, w(x, 0) = 0
    
    Parameters
    ----------
    x : np.ndarray
        Spatial grid points
        
    Returns
    -------
    v0 : np.ndarray
        Initial v (N,)
    w0 : np.ndarray
        Initial w (N,)
    """
    n = len(x)
    v0 = np.zeros(n)
    w0 = np.zeros(n)
    return v0, w0


def solve_full_model(
    n: int = 512,
    t0: float = 0.0,
    tf: float = 8.0,
    num_snapshots: int = 100,
    epsilon: float = EPSILON,
    b: float = B,
    gamma: float = GAMMA,
    c: float = C
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the full 1D FitzHugh-Nagumo PDE with Neumann boundary conditions.
    
    The system is:
        d/dt [v]   = [ε D              -1/ε I  ] [v]   + [1/ε · f(v)    ]
             [w]     [b I              -γ I    ] [w]     [0              ]
                     + [c/ε · 1  ]   + [ε/h · i_0(t) · g]
                       [c · 1    ]     [0               ]
    
    Parameters
    ----------
    n : int
        Number of spatial grid points
    t0 : float
        Initial time
    tf : float
        Final time
    num_snapshots : int
        Number of time snapshots to save
    epsilon : float
        Timescale separation parameter ε
    b : float
        Parameter b in w equation
    gamma : float
        Parameter γ in w equation
    c : float
        Constant c
        
    Returns
    -------
    x : np.ndarray
        Spatial grid (N,)
    t : np.ndarray
        Time snapshots (num_snapshots,)
    Y : np.ndarray
        Full state snapshots [v; w] (2N, num_snapshots)
    G : np.ndarray
        Nonlinear term snapshots [f(v)/ε; 0] (2N, num_snapshots)
    A : np.ndarray
        Block linear operator matrix (2N, 2N)
    """
    # Build spatial discretization
    D, h, x = build_neumann_laplacian_1d(n)
    g = boundary_input_vector(n)
    
    # Initial conditions
    v0, w0 = initial_condition(x)
    y0 = np.concatenate([v0, w0])  # [v; w] ∈ ℝ²ᴺ
    
    # Time evaluation points
    t_eval = np.linspace(t0, tf, num_snapshots)
    
    # Solve ODE system with parameters passed through lambda
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, D, g, h, n, epsilon, b, gamma, c),
        t_span=(t0, tf),
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        dense_output=False
    )
    
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    
    # Extract full state snapshots
    Y = sol.y  # (2N, num_snapshots)
    Y_v = Y[:n, :]  # (N, num_snapshots)
    
    # Build the block linear operator matrix A
    # A = [ε D          -1/ε I]
    #     [b I          -γ I  ]
    A = np.zeros((2*n, 2*n))
    A[:n, :n] = epsilon * D
    A[:n, n:] = -(1.0 / epsilon) * np.eye(n)
    A[n:, :n] = b * np.eye(n)
    A[n:, n:] = -gamma * np.eye(n)
    
    # Compute full 2N-dimensional nonlinear term snapshots
    # G = [f(v)/ε; 0]
    G = np.zeros_like(Y)
    for i in range(Y_v.shape[1]):
        G[:n, i] = nonlinear_term_fhn(Y_v[:, i]) / epsilon
    # G[n:, :] remains zero (no nonlinearity in w equation)
    
    return x, sol.t, Y, G, A
