"""
Utility mappings

"""
import numpy as np
from numpy.typing import NDArray

from numba import njit


@njit
def henon_forward(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    4D Henon symplectic mapping.

    Parameters
    ----------
    x: NDArray[np.float64], (dimension, )
        Initial condition.
    parameters: NDArray[np.float64]
        Mapping parameters ordered as [nux, nuy, mu].

    Returns
    -------
    NDArray[np.float64]

    """
    qx, qy, px, py = x
    nux, nuy, mu = parameters
    ax = 2.0*np.pi*nux
    ay = 2.0*np.pi*nuy
    cx = np.cos(ax)
    sx = np.sin(ax)
    cy = np.cos(ay)
    sy = np.sin(ay)
    Qx = cx*qx + sx*(px + qx**2 - qy**2 + mu*(qx**3 - 3.0*qx*qy**2))
    Qy = cy*qy + sy*(py - 2.0*qx*qy + mu*(-3.0*qx**2*qy + qy**3))
    Px = cx*(px + qx**2 - qy**2 + mu*(qx**3 - 3.0*qx*qy**2)) - sx*qx
    Py = cy*(py - 2.0*qx*qy + mu*(-3.0*qx**2*qy + qy**3)) - sy*qy
    return np.array([Qx, Qy, Px, Py])


@njit
def henon_inverse(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    4D Henon symplectic mapping inverse.

    Parameters
    ----------
    x: NDArray[np.float64], (dimension, )
        Initial condition.
    parameters: NDArray[np.float64]
        Mapping parameters ordered as [nux, nuy, mu].

    Returns
    -------
    NDArray[np.float64]

    """
    qx, qy, px, py = x
    nux, nuy, mu = parameters
    ax = 2.0*np.pi*nux
    ay = 2.0*np.pi*nuy
    cx = np.cos(ax)
    sx = np.sin(ax)
    cy = np.cos(ay)
    sy = np.sin(ay)
    Qx = cx*qx - sx*px
    Qy = cy*qy - sy*py
    Px = cx*px + sx*qx - Qx**2 + Qy**2 - mu*(Qx**3 - 3.0*Qx*Qy**2)
    Py = cy*py + sy*qy + 2.0*Qx*Qy - mu*(-3.0*Qx**2*Qy + Qy**3)
    return np.array([Qx, Qy, Px, Py])


@njit
def froeschle_forward(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    4D Froeschle symplectic mapping.

    Parameters
    ----------
    x: NDArray[np.float64], (dimension, )
        Initial condition.
    parameters: NDArray[np.float64]
        Mapping parameters ordered as [kx, ky, kxy].

    Returns
    -------
    NDArray[np.float64]

    """
    qx, qy, px, py = x
    kx, ky, kxy = parameters
    Qx = qx + px
    Qy = qy + py
    Px = px + kx/(2.0*np.pi)*np.sin(2.0*np.pi*Qx) + kxy*np.sin(2.0*np.pi*(Qx + Qy))
    Py = py + ky/(2.0*np.pi)*np.sin(2.0*np.pi*Qy) + kxy*np.sin(2.0*np.pi*(Qx + Qy))
    Qx = ((Qx + 0.5) % 1.0) - 0.5
    Qy = ((Qy + 0.5) % 1.0) - 0.5
    Px = ((Px + 0.5) % 1.0) - 0.5
    Py = ((Py + 0.5) % 1.0) - 0.5
    return np.array([Qx, Qy, Px, Py])


@njit
def froeschle_inverse(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    4D Froeschle symplectic mapping inverse.

    Parameters
    ----------
    x: NDArray[np.float64], (dimension, )
        Initial condition.
    parameters: NDArray[np.float64]
        Mapping parameters ordered as [kx, ky, kxy].

    Returns
    -------
    NDArray[np.float64]

    """
    Qx, Qy, Px, Py = x
    kx, ky, kxy = parameters
    px = Px - kx/(2.0*np.pi)*np.sin(2.0*np.pi*Qx) - kxy*np.sin(2.0*np.pi*(Qx + Qy))
    py = Py - ky/(2.0*np.pi)*np.sin(2.0*np.pi*Qy) - kxy*np.sin(2.0*np.pi*(Qx + Qy))
    qx = Qx - px
    qy = Qy - py
    qx = ((qx + 0.5) % 1.0) - 0.5
    qy = ((qy + 0.5) % 1.0) - 0.5
    px = ((px + 0.5) % 1.0) - 0.5
    py = ((py + 0.5) % 1.0) - 0.5
    return np.array([qx, qy, px, py])


@njit
def moser_forward(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    4D Moser symplectic mapping.

    Parameters
    ----------
    x: NDArray[np.float64], (dimension, )
        Initial condition.
    parameters: NDArray[np.float64]
        Mapping parameters [a, b, c, alpha, beta, gamma, delta, mu]
        C = [[alpha, beta], [gamma, delta]]
        U(xi) = a*xi1 + b*xi2 + 0.5*c*xi1**2 + mu*xi1**3 + xi1*xi2**2

    Returns
    -------
    NDArray[np.float64]

    """
    xi1, xi2, eta1, eta2 = x
    a, b, c, alpha, beta, gamma, delta, mu = parameters
    det = alpha*delta - beta*gamma
    dU_dxi1 = a + c*xi1 + 3.0*mu*xi1**2 + xi2**2
    dU_dxi2 = b + 2.0*xi1*xi2
    rhs1 = -eta1 + alpha*xi1 + beta*xi2 + dU_dxi1
    rhs2 = -eta2 + gamma*xi1 + delta*xi2 + dU_dxi2
    xi1p = xi1 + (delta*rhs1 - gamma*rhs2)/det
    xi2p = xi2 + (-beta*rhs1 + alpha*rhs2)/det
    eta1p = alpha*xi1 + beta*xi2
    eta2p = gamma*xi1 + delta*xi2
    return np.array([xi1p, xi2p, eta1p, eta2p])


@njit
def moser_inverse(
    x: NDArray[np.float64],
    parameters: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    4D Moser symplectic mapping inverse.

    Parameters
    ----------
    x: NDArray[np.float64], (dimension, )
        Initial condition.
    parameters: NDArray[np.float64]
        Mapping parameters [a, b, c, alpha, beta, gamma, delta, mu]
        C = [[alpha, beta], [gamma, delta]]
        U(xi) = a*xi1 + b*xi2 + 0.5*c*xi1**2 + mu*xi1**3 + xi1*xi2**2

    Returns
    -------
    NDArray[np.float64]

    """
    xi1p, xi2p, eta1p, eta2p = x
    a, b, c, alpha, beta, gamma, delta, mu = parameters
    det = alpha*delta - beta*gamma
    xi1 = (delta*eta1p - beta*eta2p)/det
    xi2 = (-gamma*eta1p + alpha*eta2p)/det
    dU_dxi1 = a + c*xi1 + 3.0*mu*xi1**2 + xi2**2
    dU_dxi2 = b + 2.0*xi1*xi2
    eta1 = eta1p + dU_dxi1 - alpha*(xi1p - xi1) - gamma*(xi2p - xi2)
    eta2 = eta2p + dU_dxi2 - beta*(xi1p - xi1) - delta*(xi2p - xi2)
    return np.array([xi1, xi2, eta1, eta2])
