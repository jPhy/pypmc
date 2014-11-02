'''This example shows how to run multiple Markov chains in parallel
with a parallel target function. MPI is used for the parallelization.

Invoke this script with the command "mpirun -n 6 python pmc_mpi.py".

'''

from __future__ import print_function
from mpi4py import MPI
import numpy as np
import pypmc
import pypmc.tools.parallel_sampler # this submodule is NOT imported by ``import pypmc``
import pypmc.tools.parallel_function # this submodule is NOT imported by ``import pypmc``

global_rank = MPI.COMM_WORLD.Get_rank()
global_size = MPI.COMM_WORLD.Get_size()

# In this example, we want to run two Markov chains.
num_parallel_samplers = 2

# Our target function has five independent parts which can be evaluated in parallel
size_parallel_functions = 3

assert global_size == num_parallel_samplers * size_parallel_functions, 'Invoke with "mpirun -n 6 python pmc_mpi.py"'

# Split the global communicator COMM_WORLD. ``likelihood_comm`` combines the processes
# [0-2] and [3-5]. The ``sampler_comm`` combines the processes [0,3], [1,4], and [2,5].
# We will only use the combinantion of the ``likelihood_comm`` master processes;
# i.e. [0,3]. Note that the global_rank = 3 process has likelihood_rank = 0 and
# sampler_rank = 1.
likelihood_comm = MPI.COMM_WORLD.Split(global_rank // size_parallel_functions)
sampler_comm = MPI.COMM_WORLD.Split(global_rank % size_parallel_functions)
likelihood_rank = likelihood_comm.Get_rank()
sampler_rank = sampler_comm.Get_rank()
if global_rank == 3:
    assert likelihood_rank == 0
    assert sampler_rank == 1


# define the target function; the target function must return the log
# of the function you want to sample from
# Here we use the product (sum on log-scale) of three Gaussians. This
# is a Gaussian again but we do not use this analytical knowlegde. Instead,
# we compute each Gaussian in a seperate process.
target_sigmas = [np.array([[0.01 , 0.0 ]
                          ,[0.0  , 0.0025]]),

                 np.array([[0.01 , 0.003 ]
                          ,[0.003, 0.0025]]),

                 np.array([[0.01 , 0.0 ]
                          ,[0.0  , 0.0025]]),
                 ]
inv_target_sigmas = [np.linalg.inv(target_sigma) for target_sigma in target_sigmas]
target_means = [np.array([4.3, 1.1]), np.array([-4.3, 3.5]), np.array([0.0, 0.0])]

def unnormalized_log_pdf_gauss(x, mu, inv_sigma):
    diff = x - mu
    return -0.5 * diff.dot(inv_sigma).dot(diff)

class ParallelGaussians(pypmc.tools.parallel_function.ParallelFunction):
    steps = 1
    def compute(self, x, rank, step):
        # ``x`` is the point in space, where the Gaussian shall be evaluated
        # ``rank`` allows to specify the work for the individual processes
        # ``step`` can be used, if intermediate results have to be broadcast; it is not important here
        return unnormalized_log_pdf_gauss(x, target_means[rank], inv_target_sigmas[rank])

log_target = ParallelGaussians(likelihood_comm)

# define a proposal
prop_dof   = 5.
prop_sigma = np.array([[0.1 , 0.  ]
                      ,[0.  , 0.02]])
prop = pypmc.density.student_t.LocalStudentT(prop_sigma, prop_dof)

# The parallelized likelihood needs some initialization and tear down, it can only
# be used in a special environment. Therefore collect all calls to ``log_target``
# into a function
def log_target_use():
    start = np.array([0., 0.])
    # instanciate the parallel Markov chain sampler
    SequentialMC = pypmc.sampler.markov_chain.AdaptiveMarkovChain
    # IMPORTANT: You MUST NOT use COMM_WORLD but ``sampler_comm`` as communicator
    parallel_sampler = pypmc.tools.parallel_sampler.MPISampler(SequentialMC, comm=sampler_comm, target=log_target, proposal=prop, start=start)

    # run and delete burn-in
    parallel_sampler.sampler.run(1000)
    parallel_sampler.clear()

    # run the parallel sampler for 10**4 steps in total with self adaptation
    for i in range(20 - 1):
        parallel_sampler.sampler.run(500)
        parallel_sampler.sampler.adapt()
    # communicate samples after last run
    parallel_sampler.run(500)

    # only the global master process collects all the samples
    if MPI.COMM_WORLD.Get_rank() == 0:
        print('Chain 0:\n%s' % parallel_sampler.history_list[0][:])
        print('Chain 1:\n%s' % parallel_sampler.history_list[1][:])

        # plot results
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('For plotting "matplotlib" needs to be installed')
            exit(1)

        both_chains_combined = np.vstack([history_item[:] for history_item in parallel_sampler.history_list])
        plt.hexbin(both_chains_combined[:,0], both_chains_combined[:,1], gridsize = 40, cmap='gray_r')
        plt.show()

# now execute the function above
log_target.use_in(log_target_use)
