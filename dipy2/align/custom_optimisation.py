
## Custom optimisation functions
# Author: Elliot Tunstall

import numpy as np

def compute_ssd_demons_step_2d(delta_field, gradient_moving, delta_field_mask, gradient_mask, sigma_sq_x, alpha=0.5, beta=0.5, out=None):
    """
    Demons step for 2D SSD-driven registration
    
    Computes the demons step for SSD-driven registration.
    
    Parameters
    ----------
    delta_field : numpy.ndarray, shape (R, C)
        The difference between the static and moving image.
    gradient_moving : numpy.ndarray, shape (R, C, 2)
        The gradient of the moving image.
    sigma_sq_x : float
        Regularization parameter.
    out : numpy.ndarray, shape (R, C, 2), optional
        Output array to store the demons step. If None, a new array will be created.
    static_mask : numpy.ndarray, shape (R, C), optional
        Mask of the static image.
    moving_mask : numpy.ndarray, shape (R, C), optional
        Mask of the moving image.
    
    Returns
    -------
    out : numpy.ndarray
        The computed demons step.
    total_energy : float
        The current SSD energy before applying the returned demons step.
    """
    if out is None:
        out = np.zeros_like(delta_field, shape=(delta_field.shape[0], delta_field.shape[1], 2))

    energy = 0
    nr, nc = delta_field.shape

    for i in range(nr):
        for j in range(nc):
            delta = delta_field[i, j]
            delta_mask = delta_field_mask[i, j]
            delta_2 = delta ** 2
            delta_2_mask = delta_mask ** 2
            energy += (alpha * delta_2) + (beta * delta_2_mask)
            nrm2 = gradient_moving[i, j, 0] ** 2 + gradient_moving[i, j, 1] ** 2
            nrm2_mask = gradient_mask[i, j, 0] ** 2 + gradient_mask[i, j, 1] ** 2
            den = delta_2 / sigma_sq_x + nrm2
            den_mask = delta_2_mask / sigma_sq_x + nrm2_mask

            if den < 1e-9 or den_mask < 1e-9:
                out[i, j, 0] = 0
                out[i, j, 1] = 0
            else:
                out[i, j, 0] = alpha * (delta * gradient_moving[i, j, 0] / den) + beta * (delta_mask * gradient_mask[i, j, 0] / den_mask)
                out[i, j, 1] = alpha * (delta * gradient_moving[i, j, 1] / den) + beta * (delta_mask * gradient_mask[i, j, 1] / den_mask)

    # # Assuming you have a function dice_coefficient to compute Dice score
    # dice_score = dice_coefficient(static_mask, moving_mask) if static_mask is not None and moving_mask is not None else 0
    
    # total_energy = alpha * energy + beta * (1 - dice_score)

    return out, energy

def dice_coefficient(static_mask, moving_mask):
    """
    Calculate the Dice coefficient between two binary masks.
    """
    intersection = np.sum(static_mask * moving_mask)
    total = np.sum(static_mask) + np.sum(moving_mask)
    return 2 * intersection / total if total > 0 else 0


def minimise(delta_field, gradient_moving, sigma_sq_x):
    """
    Minimise the energy function for SSD-driven registration.

    Parameters
    ----------
    delta_field : numpy.ndarray, shape (R, C)
        The difference between the static and moving image.
    gradient_moving : numpy.ndarray, shape (R, C, 2)
        The gradient of the moving image.
    sigma_sq_x : float
        Regularization parameter.
    
    Returns
    -------
    den : float
        The denominator for gradient based optimisation.
    """
    nr, nc = delta_field.shape

    for i in range(nr):
        for j in range(nc):
            delta = delta_field[i, j]
            delta_2 = delta ** 2
            energy += delta_2
            nrm2 = gradient_moving[i, j, 0] ** 2 + gradient_moving[i, j, 1] ** 2
            den = delta_2 / sigma_sq_x + nrm2

    return den, delta
