'''Unit tests for parallel sampler
In order to run tests in parallel, you have to execute this test with
"mpirun", for example: "mpirun -n 2 nosetests parallel_sampler_test.py"

'''

import numpy as np
import unittest
from .parallel_function import ParallelFunction

def setUpModule():
    try:
        from mpi4py import MPI
    except ImportError:
        raise unittest.SkipTest("Cannot test MPI parallelism without MPI4Py")

    from .parallel_sampler import MPISampler, MPI

    global MPI
    global MPISampler
    global comm
    global global_rank
    global global_size
    comm = MPI.COMM_WORLD
    global_rank = comm.Get_rank()
    global_size = comm.Get_size()

class TestParallelFunction(unittest.TestCase):
    def test_error_messages(self):
        parallel_function = ParallelFunction()

        self.assertRaisesRegexp(AssertionError, '.*ParallelFunction.*only.*self.use_in', parallel_function, np.array([9,3.4,6.6]))

        def foo():
            parallel_function(np.array([9,3.4,6.6]))

        with self.assertRaisesRegexp(NotImplementedError, '.*compute.*overridden.*by user'):
            parallel_function.use_in(foo)

    def test_compute_single_step(self):
        class MyParallelFunction(ParallelFunction):
            def compute(self, worker_input, rank, step):
                return worker_input[rank]

        parallel_function = MyParallelFunction()

        def my_parallel_env():
            parallel_out = parallel_function([1.] * global_size)
            self.assertAlmostEqual(parallel_out, global_size, places=13)
            parallel_out = parallel_function([2.] * global_size)
            self.assertAlmostEqual(parallel_out, 2.0*global_size, places=13)

        parallel_function.use_in(my_parallel_env)

    def test_compute_multiple_steps(this_unittest):
        ran_compute_step = [False, False]
        ran_combine_step = [False, False]

        class MyParallelFunction(ParallelFunction):
            steps = 2
            def compute(this_parallel_function, worker_input, rank, step):
                if step == 0:
                    ran_compute_step[0] = True
                    return worker_input[rank]
                elif step == 1:
                    if global_rank == 0:
                        this_unittest.assertTrue(ran_combine_step[0])
                    ran_compute_step[1] = True
                    this_unittest.assertEqual(worker_input, (1.0, 5)) # this is the output of combine(..., step=0)
                    return worker_input[0] + 1 # 2.0
                else:
                    raise RuntimeError('Should only run two steps')

            def combine(this_parallel_function, worker_output, step):
                if step == 0:
                    this_unittest.assertTrue(ran_compute_step[0])
                    ran_combine_step[0] = True
                    this_unittest.assertEqual(worker_output, [1.] * global_size)
                    return (np.prod(worker_output), 5)
                elif step == 1:
                    this_unittest.assertTrue(ran_compute_step[1])
                    ran_combine_step[1] = True
                    this_unittest.assertEqual(worker_output, [2.0] * global_size)
                    return 10
                else:
                    raise RuntimeError('Should only run two steps')

        def my_parallel_env():
            parallel_out = parallel_function([1.] * global_size)
            this_unittest.assertEqual(parallel_out, 10)

        parallel_function = MyParallelFunction()
        parallel_function.use_in(my_parallel_env)

        # ``combine`` is only executed by the main process
        if global_rank == 0:
            this_unittest.assertTrue(ran_combine_step[0])
            this_unittest.assertTrue(ran_combine_step[1])
        this_unittest.assertTrue(ran_compute_step[0])
        this_unittest.assertTrue(ran_compute_step[1])

    def test_default_combine(self):
        class MyParallelFunction(ParallelFunction):
            def compute(this_parallel_function, worker_input, rank, step):
                return worker_input[rank]

        def my_parallel_env():
            parallel_out = parallel_function([1.] * global_size)
            self.assertAlmostEqual(parallel_out, global_size, places=10)

        parallel_function = MyParallelFunction()
        parallel_function.use_in(my_parallel_env)

    def test_return_of_use_in(self):
        class MyParallelFunction(ParallelFunction):
            def compute(this_parallel_function, worker_input, rank, step):
                return worker_input[rank]

        parallel_function = MyParallelFunction()

        @parallel_function.use_in
        def fourty_two():
            return 42

        # return value only present in main process
        if global_rank == 0:
            self.assertEqual(fourty_two, 42)

        # ``parallel_function.use_in`` should not call exit
        # test if all processes arrive here
        communicated_range = comm.allgather(global_rank)
        # list(range(...)) because in python3 type(range(...)) != list
        std_range = list(range(global_size))
        self.assertEqual(communicated_range, std_range)
