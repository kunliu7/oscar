import numpy as np
from scipy.fftpack import idct

from .compressed_sensing import recon_2D_by_LASSO, recon_2D_by_cvxpy



def recon_4D_landscape_by_2D(
    origin: np.ndarray,
    sampling_frac: float,
    random_indices: np.ndarray=None,
    method: str="BP"
) -> np.ndarray:
    """Reconstruct landscapes by sampling on given landscapes.

    """
    # ! Convention: First beta, Last gamma
    shape_4d = origin.shape
    origin_2d = origin.reshape(shape_4d[0] * shape_4d[1], 
        shape_4d[2] * shape_4d[3])
    
    ny, nx = origin_2d.shape

    n_pts = np.prod(shape_4d)

    print(f"total samples: {n_pts}")

    # extract small sample of signal
    k = round(n_pts * sampling_frac)
    if not isinstance(random_indices, np.ndarray):
        ri = np.random.choice(n_pts, k, replace=False) # random sample of indices
    else:
        print("use inputted random indices")
        assert len(random_indices.shape) == 1 and random_indices.shape[0] == k
        ri = random_indices # for short 

    # create dct matrix operator using kron (memory errors for large ny*nx)
    # idct_list = [idct(np.identity(dim), norm='ortho', axis=0) for dim in shape]
    # A = idct_list[0]
    # for i in range(1, 4):
    #     A = np.kron(A, idct_list[i])

    A = np.kron(
        idct(np.identity(nx), norm='ortho', axis=0),
        idct(np.identity(ny), norm='ortho', axis=0),
    )
    A = A[ri,:] # same as phi times kron

    # b = X.T.flat[ri]
    # recon = recon_4D_by_cvxpy(shape, A, origin.T.flat[ri])
    b = origin_2d.T.flat[ri]
    if method == 'BP':
        recon = recon_2D_by_cvxpy(nx, ny, A, b)
    elif method == 'BPDN':
        recon = recon_2D_by_LASSO(nx, ny, A, b, 0.001)
    else:
        assert False, "Invalid CS method"

    recon = recon.reshape(*shape_4d)

    print('end: solve l1 norm')
    return recon

