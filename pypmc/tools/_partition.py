'''Implements the "minimal lexicographic integer partition"

'''

import numpy as _np
from pypmc.density.gauss import Gauss
from pypmc.density.mixture import MixtureDensity

def partition(N, k):
    '''Distributre ``N`` into ``k`` partitions such that each partition
    takes the value ``N//k`` or ``N//k + 1`` where ``//`` denotes integer
    division.

    Example: N = 5, k = 2  -->  return [3, 2]

    '''
    out = [N // k] * k
    remainder = N % k
    for i in range(remainder):
        out[i] += 1
    return out

def patch_data(data, L=100, try_diag=True):
    '''Patch ``data`` (for example Markov chain output) into parts of
    length ``L``. Return a Gaussian mixture where each component gets
    the empirical mean and covariance of one patch.

    :param data:

        Matrix-like array; the points to be patched. Expect ``data[i]``
        as the d-dimensional i-th point.

    :param L:

        Integer; the length of one patch. The last patch will be shorter
        if ``L`` is not a divisor of ``len(data)``.

    :param try_diag:

        Bool; If some patch does not define a proper covariance matrix,
        it cannot define a Gaussian component. ``try_diag`` defines how
        to handle that case:
        If ``True`` (default), the off-diagonal elements are set to zero
        and it is tried to form a Gaussian with that matrix again. If
        that fails as well, the patch is skipped.
        If ``False`` the patch is skipped directly.

    '''
    # patch data into length L patches
    patches = _np.array([data[patch_start:patch_start + L] for patch_start in range(0, len(data), L)])

    # calculate means and covs
    means   = _np.array([_np.mean(patch,   axis=0) for patch in patches])
    covs    = _np.array([_np.cov (patch, rowvar=0) for patch in patches])

    # form gaussian components
    components = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        try:
            this_comp = Gauss(mean, cov)
            if this_comp.det_sigma <= 0:
                raise _np.linalg.LinAlgError('Negative determinant.')
            components.append(this_comp)
        except _np.linalg.LinAlgError as error1:
            print("Could not form Gauss from patch %i. Reason: %s" %(i,repr(error1)))
            if try_diag:
                cov = _np.diag(_np.diag(cov))
                try:
                    this_comp = Gauss(mean, cov)
                    if this_comp.det_sigma <= 0:
                        raise _np.linalg.LinAlgError('Negative determinant.')
                    components.append(this_comp)
                    print('Diagonal covariance attempt succeeded.')
                except _np.linalg.LinAlgError as error2:
                    print("Diagonal covariance attempt failed. Reason: %s" %repr(error2))

    # create and return mixture
    return MixtureDensity(components)
