import numpy as np
from numpy.typing import NDArray

from numba import njit


@njit
def forward2D(
    x: NDArray[np.float64], 
    parameters: NDArray[np.float64]
) ->  NDArray[np.float64]:
    """
    Sample 2D symplectic mapping (forward)
    Henon mapping (McMillan form)

    """
    q, p = x
    a, b = parameters
    Q = p
    P = -q + a*p + (1 - b)*p**2 + b*p**3
    return np.stack((Q, P))


@njit
def inverse2D(
    x: NDArray[np.float64], 
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Sample 2D symplectic mapping (inverse)
    Henon mapping (McMillan form)

    """
    q, p = x
    a, b = parameters
    Q = -p + a*q + (1 - b)*q**2 + b*q**3
    P = q
    return np.stack((Q, P))


@njit
def forward4D(
    x: NDArray[np.float64], 
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Sample 4D symplectic mapping (forward)
    Accelerator mapping (4D Henon)

    """
    qx, qy, px, py = x
    cx, sx, cy, sy, mu = parameters
    Qx = cx*qx + sx*(px + qx**2 - qy**2 + mu*(qx**3 - 3*qx*qy**2))
    Qy = cy*qy + sy*(py - 2*qx*qy + mu*(-3*qx**2*qy + qy**3))
    Px = cx*(px + qx**2 - qy**2 + mu*(qx**3 - 3*qx*qy**2)) - sx*qx
    Py = cy*(py - 2*qx*qy + mu*(-3*qx**2*qy + qy**3)) - sy*qy
    return np.stack((Qx, Qy, Px, Py))


@njit
def inverse4D(
    x: NDArray[np.float64], 
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Sample 4D symplectic mapping (inverse)
    Accelerator mapping (4D Henon)

    """
    qx, qy, px, py = x
    cx, sx, cy, sy, mu = parameters
    Qx = cx*qx - sx*px
    Qy = cy*qy - sy*py
    Px = cx*px + sx*qx - Qx**2 + Qy**2 - mu*(Qx**3 - 3*Qx*Qy**2)
    Py = cy*py + sy*qy + 2*Qx*Qy - mu*(-3*Qx**2*Qy + Qy**3)
    return np.stack((Qx, Qy, Px, Py))


@njit
def bb_map_forward(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64],
    epsilon: float = 1.0e-18
) -> NDArray[np.float64]:
    """
    Beam-beam map (forward)
    
    """
    q, p = x + epsilon
    nu, xi, ks = parameters
    c = np.cos(2.0*np.pi*nu)
    s = np.sin(2.0*np.pi*nu)
    Q = p
    P = -q + 2.0*c*p + 8.0*np.pi*xi*s/p*(np.exp(-0.5*p*p) - 1.0) + s*ks*p*p
    return np.stack((Q, P))


@njit
def bb_map_inverse(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64],
    epsilon: float = 1.0e-18
) -> NDArray[np.float64]:
    """
    Beam-beam map (inverse)
    
    """
    q, p = x + epsilon
    nu, xi, ks = parameters
    c = np.cos(2.0*np.pi*nu)
    s = np.sin(2.0*np.pi*nu)
    Q = -p + 2.0*c*q + 8.0*np.pi*xi*s/q*(np.exp(-0.5*q*q) - 1.0) + s*ks*q*q
    P = q
    return np.stack((Q, P))


@njit
def bb_map_diagonal_symmetry(
    q: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Diagonal symmetry line
    
    """
    nu, xi, ks = parameters
    p = q
    return p


@njit
def bb_map_force_symmetry(
    q: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Force symmetry line
    
    """
    nu, xi, ks = parameters
    c = np.cos(2.0*np.pi*nu)
    s = np.sin(2.0*np.pi*nu)    
    p = 0.5*(2.0*q*c + (ks*q*q + (8.0*np.pi*xi)*(np.exp(-0.5*q*q) - 1.0)/q)*s)
    return p
