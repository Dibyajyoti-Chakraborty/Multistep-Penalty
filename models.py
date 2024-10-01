import jax
import jax.numpy as jnp
import numpy as np
import flax
import flax.linen as nn

class Encoder(nn.Module):
    c_hid : int
    latent_dim : int

    @nn.compact
    def __call__(self, x):
        ###--------------- Encoder --------------------
        x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),strides=(2,2),padding='same')(x)
        x = nn.gelu(x) # relu(x) #
        x = nn.Conv(features=self.latent_dim,  kernel_size=(3, 3) ,strides=(2,2),padding='same')(x)
        return x
    
class Decoder(nn.Module):
    c_out : int
    c_hid : int
    latent_dim : int

    @nn.compact
    def __call__(self, x):
        
        ###--------------- Decoder --------------------
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3),strides=(2,2),padding='SAME')(x)
        x = nn.gelu(x) # relu(x) #
        x = nn.Conv(features=self.c_out,  kernel_size=(3, 3),padding='SAME')(x)
        x = nn.gelu(x) # relu(x) #
        x = nn.ConvTranspose(features=self.c_out,  kernel_size=(3, 3), strides=(2,2),padding='SAME')(x)
        x = nn.gelu(x) # relu(x) #
        x = nn.Conv(features=self.c_out,  kernel_size=(3, 3),padding='SAME')(x)
        return x
    
class CNN(nn.Module):
    c_hid : int
    latent_dim : int
    
    @nn.compact
    def __call__(self, x):
        ###---------------dilated CNN -------------------- 
        for i in range(4):
            input = x
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=1)(x) 
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=2)(x) 
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=3)(x)  
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=4)(x) 
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=8)(x)
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=16)(x)
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=32)(x)
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=16)(x)
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=8)(x)
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=4)(x)
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=3)(x)
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.c_hid,  kernel_size=(3, 3),kernel_dilation=2)(x)
            x = nn.gelu(x) # relu(x) #
            x = nn.Conv(features=self.latent_dim,  kernel_size=(3, 3))(x)
            x = nn.gelu(x) # relu(x) #
            x = input + x
        return x

class dil_CNN(flax.struct.PyTreeNode):
    encoder: nn.Module
    d_cnn: nn.Module
    decoder: nn.Module
    
    def init(self, coords):
        rng, encoder_rng, d_cnn_rng, decoder_rng = jax.random.split(jax.random.PRNGKey(np.random.randint(10)), 4)
        coords, encoder_params = self.encoder.init_with_output(encoder_rng, coords)
        coords, d_cnn_params = self.d_cnn.init_with_output(d_cnn_rng, coords)
        coords, decoder_params = self.decoder.init_with_output(decoder_rng, coords)
        return {
            "encoder": encoder_params,
            "d_cnn": d_cnn_params,
            "decoder": decoder_params
        }
        
    def apply(self, params, inp):
        x = inp
        enc = self.encoder.apply(params["encoder"], inp)
        layers = self.d_cnn.apply(params["d_cnn"],enc)
        pred = self.decoder.apply(params["decoder"], layers)
        return x + pred # 

class MP_CNN(flax.struct.PyTreeNode):
    ''' 
    Multi-step Penalty optimization : Chakraborty et. al. (https://arxiv.org/abs/2407.00568)
    '''
    encoder: nn.Module
    d_cnn: nn.Module
    decoder: nn.Module
    rollouts : int = 2 # number of rollouts before a split
    n_splits: int = 3 # number of discontinuities(splits)
    
    def init(self, coords):
        rng, encoder_rng, d_cnn_rng, decoder_rng = jax.random.split(jax.random.PRNGKey(np.random.randint(10)), 4)
        coords, encoder_params = self.encoder.init_with_output(encoder_rng, coords)
        coords, d_cnn_params = self.d_cnn.init_with_output(d_cnn_rng, coords)
        coords, decoder_params = self.decoder.init_with_output(decoder_rng, coords)
        return {
            "encoder": encoder_params,
            "d_cnn": d_cnn_params,
            "decoder": decoder_params,
            "del_q": jnp.zeros((self.n_splits,coords.shape[0],coords.shape[1],coords.shape[2]))
        }
        
    def apply(self, params, inp):
        x = inp
        preds = []
        # Total rollouts is rollouts*(n_splits + 1)
        for i in range(self.n_splits+1):
            for _ in range(self.rollouts):
                enc = self.encoder.apply(params["encoder"], x)
                layers = self.d_cnn.apply(params["d_cnn"],enc)
                x = (x + 1.0*self.decoder.apply(params["decoder"], layers)) # use 0.1 if the chnage between timesteps is small in data
                preds.append(x)
            if i<self.n_splits:
                # use stop gradients to save memory : Pushforward trick
                x = jax.lax.stop_gradient(x) + (params['del_q'][i])
        return jnp.array(preds)


