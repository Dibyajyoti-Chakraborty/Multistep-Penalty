import jax
import jax.numpy as jnp
import numpy as np
from models import *


def Dataloader(data,batch_size,timesteps):
    '''
    Convert shape from :       total_snapshots x H x W x ch -> n_batches x batch_size x timesteps x H x W x ch
    '''
    time_chunks = []
    for i in range(data.shape[0] - timesteps):
        time_chunks.append(data[i:i+timesteps])
    extra = len(time_chunks) % batch_size
    time_chunks = np.array(time_chunks[:-extra])
    split = np.random.permutation(np.array(np.split(time_chunks,time_chunks.shape[0]//batch_size)))
    return split