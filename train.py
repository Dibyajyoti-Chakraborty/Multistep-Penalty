import jax
import jax.numpy as jnp
import numpy as np
import flax
import argparse
import optax
import pickle
from flax.training.common_utils import shard
from functools import partial
from models import *
from utils import *

n_devices = jax.device_count()
print(n_devices)

def _train(train_data,
           test_data,
           lr,epochs,
           name,
           load=False):
    
    lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=lr,
            peak_value=lr,
            warmup_steps=0.1*epochs*train_data.shape[0],
            decay_steps=0.9*epochs*train_data.shape[0],
            end_value=lr/10.
        ) 
    model = MP_CNN(
        encoder = Encoder(c_hid = 32,latent_dim = 32),
        decoder = Decoder(c_out = 2, c_hid =32, latent_dim = 32 ),
        d_cnn = CNN(c_hid = 32,latent_dim = 32),
        rollouts=rollouts,
        n_splits=n_splits,
                   )
    
    params = model.init(train_data[0,:int(train_data.shape[1]/n_devices),0])
    optimizer = optax.adam(learning_rate = lr_scheduler)

    
    def loss(params,batch,mu):
        preds = model.apply(params,batch[:,0])
        L_GT = jnp.mean((jnp.abs(preds[:,1:] - batch[:,1:]))) 
        L_penalty = jnp.mean(jnp.abs(params['del_q']))
        return jnp.mean(L_GT+ mu*L_penalty)
    gloss = lambda params,batch,mu :jax.jit(jax.value_and_grad(jax.jit(loss)))(params,batch,mu)
    
    if load==True:
        params = pickle.load(open(f'params/{name}','rb'))
        print("Loaded")
    best_loss = loss(params,test_data[np.random.randint(len(test_data))],0)
    print("Start Model Loss:",best_loss)
    print("Parameters:",sum(x.size for x in jax.tree_leaves(params)))
    opt_state = optimizer.init(params)

    # Multi GPU training
    @partial(jax.pmap, axis_name='device')
    def step(params,opt_state,batch,mu):
        loss_value , grads = gloss(params,batch,mu)
        grads = jax.lax.pmean(grads, axis_name='device')
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_value = jax.lax.pmean(loss_value, axis_name='device')
        return params, opt_state, loss_value
    
    params = flax.jax_utils.replicate(params)
    opt_state = flax.jax_utils.replicate(opt_state)
    mu_one = jnp.array([1]) # change mu after certain number of steps - hyperparameter.
    mu = flax.jax_utils.replicate(mu_one)
    for i in range(1,epochs+1): 
        losses = []
        for batch in train_data:
            batch = shard(batch)
            params, opt_state, loss_value = step(params,opt_state,batch,mu) 
            loss_value = flax.jax_utils.unreplicate(loss_value)
            losses.append(loss_value)
        net_loss = np.mean(np.array(losses))
        local_params = flax.jax_utils.unreplicate(params)
        test_loss = []
        for test_batch in test_data:
            test_loss.append(loss(local_params,test_batch,mu_one))
        test_loss = np.mean(np.array(test_loss))
        if test_loss < best_loss:
            best_loss = test_loss
            pickle.dump(local_params,open(f'params/{name}','wb'))
            print("Saved!!!")
        P = jnp.mean(jnp.abs(local_params['del_q']))
        print(f'Epoch : {i}, mu : {flax.jax_utils.unreplicate(mu)} Train Loss : {net_loss}, Test Loss : {test_loss}, Penalty Loss : {P}')

if __name__ == '__main__':
    rollouts = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, help="number of splits",
                        nargs='?', default=3, const=0)
    parser.add_argument("--lr", type=float, help="Learning rate",
                        nargs='?', default=1e-4, const=0)
    parser.add_argument("--batch_size", type=int, help="Batch size",
                        nargs='?', default=32, const=0)
    args = parser.parse_args()
    n_splits = args.n_splits
    lr = args.lr
    batch_size = args.batch_size
    
    n_step = (rollouts*(n_splits+1)+1)
    name = 'mp_model'
    timesteps = np.arange(0,n_step)
    print(timesteps)
    data = np.load("data.npy")
    print(data.shape)
    train_limit = int(0.8*data.shape[0])
    train_data = Dataloader(data[:train_limit],batch_size=batch_size,batch_time = len(timesteps))
    print(train_data.shape)
    test_data = Dataloader(data[train_limit:],batch_size=batch_size,batch_time = len(timesteps))
    print(test_data.shape)
    del data
    _train(train_data, test_data, lr=lr,epochs=500,name=name,load=False)