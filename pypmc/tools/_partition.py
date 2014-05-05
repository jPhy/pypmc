'''Implements the "minimal lexicographic integer partition"

'''

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
