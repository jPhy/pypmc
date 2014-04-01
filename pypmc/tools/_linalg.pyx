cdef double bilinear_sym(_np.ndarray[double, ndim=2] matrix, _np.ndarray[double, ndim=1] vector):
    '''Compute the bilinear form of a *symmetric* matrix and a vector.

    Return
    .. math::

           x^T \cdot M \cdot x

    where :math:`M=M^T`.

    .. warning::

           No checking is performed on the size of the inputs, or if
           the matrix indeed is symmetric. Expect segfault if
           assumptions are violated.

    '''

    cdef:
        double res = 0.0
        size_t i, j

    for i in range(len(vector)):
        # diagonal contribution
        res += vector[i] * vector[i] * matrix[i,i]
        for j in range(i):
            # off-diagonal elements come twice
            for j in range(i):
                res += 2. * vector[i] * vector[j] * matrix[i,j]

    return res
