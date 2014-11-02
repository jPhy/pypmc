'''Interface for parallel evaluable function

'''

from mpi4py import MPI
import numpy as _np

class _StopWorkerMainloop(object):
    pass

class ParallelFunction(object):
    '''A callable wich is executed in multiple processes. This abstract
    base class provides an Interface for easier parallel programming.
    The user should derive from this class and override the method
    :py:meth:`ParallelFunction.compute` and
    :py:meth:`ParallelFunction.combine` if desired. If the function needs
    broadcasting of intermediate results, the member variable steps can
    be used, otherwise it may safely be ignored.

    :param comm:

        ``mpi4py`` communicator; the communicator to be used for internal
        communication.

    '''
    steps = 1

    def __init__(self, comm=MPI.COMM_WORLD):
        self.comm = comm
        self.rank = comm.Get_rank()
        self._in_use = False

    def compute(self, worker_input, rank, step):
        '''User defined function. This function is executed by all workers
        with their ``rank`` and the current ``step`` as arguments.
        ``worker_input`` is the argument passed to
        :py:meth:`ParallelFunction.__call__` in ``step`` zero and the
        output of :py:meth:`ParallelFunction.combine` in all other steps.

        '''
        raise NotImplementedError('``compute`` must be overridden by user')

    def combine(self, worker_output, step):
        '''User definable function. This function is executed by the main
        process. By default all worker outputs are summed.

        '''
        return _np.sum(worker_output)

    def use_in(self, function, *args, **kwargs):
        '''Use the :py:class:`ParallelFunction` inside the ``function``.
        A :py:class:`ParallelFunction` needs special intialization and tear
        down. For safe usage, it is only allowed to pass an external function
        which calls the :py:class:`ParallelFunction` to
        :py:meth:`ParallelFunction.use_in`.

        :param fucntion:

            callable; the function to be executed. Inside that function,
            the invoking :py:class:`ParallelFunction` is callable. The
            return value of ``function`` is forwarded.

        :param args, kwargs:

            additional arguments to be passed to ``function``.

        '''
        self._in_use = True
        if self.rank == 0:
            func_out = None
            try:
                func_out = function(*args, **kwargs)
            finally:
                self._break_worker_mainloop()
                self._in_use = False
            return func_out
        else:
            self._worker_mainloop()
            self._in_use = False

    def __call__(self, x):
        assert self._in_use, 'A ``ParallelFunction`` can only be called using ``self.use_in``.'
        self.comm.bcast(x)
        for step in range(self.steps - 1):
            x = self.comm.gather(self.compute(x, self.rank, step))
            x = self.comm.bcast(self.combine(x, step))
        worker_output = self.comm.gather(self.compute(x, self.rank, self.steps - 1))
        return self.combine(worker_output, self.steps - 1)

    def _worker_mainloop(self):
        """send worker processes into main loop"""
        assert self.rank != 0

        while True:
            x = self.comm.bcast()
            if x is _StopWorkerMainloop:
                break
            for step in range(self.steps - 1):
                self.comm.gather(self.compute(x, self.rank, step))
                x = self.comm.bcast()
            self.comm.gather(self.compute(x, self.rank, self.steps - 1))

    def _break_worker_mainloop(self):
        self._in_use = False
        if self.rank == 0:
            self.comm.bcast(_StopWorkerMainloop)
