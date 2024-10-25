import jax.numpy as jnp
from jax import random
from flax import linen as nn
import other
from jax import scipy as jsp    


num_units = 512
# class OrthogMat(nn.Module):
#     num_evecs: int  # Dimension of the matrix
    
#     def setup(self):
#         # Parametrize the eigenvectors
#         self.eigenvectors = self.param('eigenvectors', nn.initializers.orthogonal(), (self.num_evecs, self.num_evecs))
#         # Parametrize the eigenvalues (can be learned or fixed)
#         self.eigenvalues = self.param('eigenvalues', nn.initializers.uniform(), (self.num_evecs,))

#     def __call__(self, x):
#         # Orthogonalize the eigenvectors (e.g., using QR decomposition)
#         # q, r = jnp.linalg.qr(self.eigenvectors)
#         # # Create diagonal matrix from eigenvalues
#         # Lambda = jnp.diag(self.eigenvalues)
#         # # Construct the orthogonal matrix
#         # C = q @ Lambda @ q.T
#         return jnp.matmul(self.eigenvectors, x)

# class OrthogMat(nn.Module):
#     num_evecs: int  # Dimension of the matrix
    
#     def setup(self):
#         # Parametrize the eigenvectors
#         self.eigenvectorsU = self.param('eigenvectorsU', nn.initializers.uniform(), (self.num_evecs, self.num_evecs))
#         self.eigenvectorsV = self.param('eigenvectorsV', nn.initializers.uniform(), (self.num_evecs, self.num_evecs))

#         # Parametrize the eigenvalues (can be learned or fixed)
#         self.eigenvalues = self.param('eigenvalues', nn.initializers.uniform(), (self.num_evecs,))

#     def __call__(self, x):
#         # Orthogonalize the eigenvectors (e.g., using QR decomposition)
#         # q, r = jnp.linalg.qr(self.eigenvectors)
#         # # Create diagonal matrix from eigenvalues
#         # Lambda = jnp.diag(self.eigenvalues)
#         # # Construct the orthogonal matrix
#         # C = q @ Lambda @ q.T
#         return jnp.matmul(self.eigenvectorsU, x),  nn.sigmoid(self.eigenvalues), self.eigenvectorsV
    

class OrthogMat(nn.Module):
    num_evecs: int  # Dimension of the matrix
    
    def setup(self):
        # Parametrize the eigenvectors
        self.eigenvectorsU = self.param('eigenvectorsU', nn.initializers.uniform(), (self.num_evecs, self.num_evecs))
        # self.eigenvectorsV = self.param('eigenvectorsV', nn.initializers.uniform(), (self.num_evecs, self.num_evecs))

        # Parametrize the eigenvalues (can be learned or fixed)
        self.eigenvalues = self.param('eigenvalues', nn.initializers.uniform(), (self.num_evecs,))

    def __call__(self, x):
        # Orthogonalize the eigenvectors (e.g., using QR decomposition)
        # q, r = jnp.linalg.qr(self.eigenvectors)
        # # Create diagonal matrix from eigenvalues
        # Lambda = jnp.diag(self.eigenvalues)
        # # Construct the orthogonal matrix
        # C = q @ Lambda @ q.T
        U = jnp.triu(self.eigenvectorsU, k=1)
        A = U - jnp.transpose(U)
        O = jnp.matmul(jnp.eye(self.num_evecs) + A, jnp.linalg.inv(jnp.eye(self.num_evecs) - A))
        # O_U = jsp.linalg.expm(A)
        # V = jnp.triu(self.eigenvectorsV, k=1)
        # B = U - jnp.transpose(V)
        # O_V = jsp.linalg.expm(B)
        # O_V = jnp.matmul(jnp.eye(self.num_evecs) + B, jnp.linalg.inv(jnp.eye(self.num_evecs) - B))
        return jnp.matmul(O, x)
        # return jnp.matmul(O_U, x),  nn.sigmoid(self.eigenvalues), O_V
    

class Encoder1(nn.Module):
  # number of latent variables in the model
  # defining like this means latents is an input parameter
 latents1: int
  #the __call__ bit means that if you define
  # encoder = Encoder()
  # and then call enconder() this will use the function defined by __call__
  # and can use parameters

 @nn.compact
 def __call__(self, x):
            #outputs 500 features
        # to input x, apply the first dense layer which we call 'fc1'
        # why is the x appearing after that?
        # we apply a dense 500 feature transformation along the last dimension of x
      x = nn.Dense(num_units, name='fc1')(x)
        #then to x apply the activation function
      x = nn.relu(x) #activation layer
      x = nn.Dense(num_units, name='fc1b')(x)
        #then to x apply the activation function
      x = nn.relu(x) #activation layer
        #on this layer, apply another layer, with inputed number of features - number of latent variables we are assuming
        #this layer is called 'fc2_mean2'
        # same thing, creating another layer for the log variance
        # again with no of outputs equal to the number of latent variables we are assuming
      mean_x = nn.Dense(self.latents1, name='fc2_mean')(x)
      logvar_x = nn.Dense(self.latents1, name='fc2_logvar')(x)
      return mean_x, logvar_x

class Encoder2(nn.Module):
  # number of latent variables in the model
  # defining like this means latents is an input parameter
 latents2: int
  #the __call__ bit means that if you define
  # encoder = Encoder()
  # and then call enconder() this will use the function defined by __call__
  # and can use parameters

 @nn.compact
 def __call__(self, x):
            #outputs 500 features
        # to input x, apply the first dense layer which we call 'fc1'
        # why is the x appearing after that?
        # we apply a dense 500 feature transformation along the last dimension of x
      x = nn.Dense(num_units, name='fc1')(x)
        #then to x apply the activation function
      x = nn.relu(x) #activation layer
      x = nn.Dense(num_units, name='fc1b')(x)
        #then to x apply the activation function
      x = nn.relu(x) #activation layer
        #on this layer, apply another layer, with inputed number of features - number of latent variables we are assuming
        #this layer is called 'fc2_mean2'
        # same thing, creating another layer for the log variance
        # again with no of outputs equal to the number of latent variables we are assuming
      mean_x = nn.Dense(self.latents2, name='fc2_mean')(x)
      logvar_x = nn.Dense(self.latents2, name='fc2_logvar')(x)
      return mean_x, logvar_x

class Decoder1(nn.Module):
  no_out1: int
    #this doesn't take any inputs because they will be implicit from the input
  @nn.compact
  def __call__(self, z):
        # the input now is z the latent variable distributions
      z = nn.Dense(num_units, name='fc1')(z)
        #activation layer
      z = nn.relu(z)
      z = nn.Dense(num_units, name='fc1b')(z)
        #activation layer
      z = nn.relu(z)
        # apply another transformation this time with 784 features not sure why?
        # might be the input size for the particular example
      logits = nn.Dense(self.no_out1, name='fc5')(z)
      return logits


class Decoder2(nn.Module):
    no_out2: int
    #this doesn't take any inputs because they will be implicit from the input
    @nn.compact
    def __call__(self, z):
        # the input now is z the latent variable distributions
        z = nn.Dense(num_units, name='fc1')(z)
        #activation layer
        z = nn.relu(z)
        z = nn.Dense(num_units, name='fc1b')(z)
        #activation layer
        z = nn.relu(z)
        # apply another transformation this time with 784 features not sure why?
        # might be the input size for the particular example
        logits = nn.Dense(self.no_out2, name='fc5')(z)
        return logits

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + jnp.multiply(eps, std)


class VAE(nn.Module):
 latents: tuple =(int,int)
 no_out: tuple =(int,int)
 alpha: float = 0.95
#  latents2: int = 20
  # defining like this means latents is an input parameter

    # this intialises variables or submodules (are they the same thing?) in the function
    # it's not a variable in the way I'd usually think of it
    # it's a small part of the overall model (module), i.e. sub - modules
  # this part is automatically done when VAE is called as
 def setup(self):
    # we call the model Encoder with the specified number of latent features.
    # because of how it was set up, this automatically calls the __call__ method we defined
   self.encoder1 = Encoder1(self.latents[0])
    # set up the decoder submodule as well
   self.decoder1 = Decoder1(self.no_out[0])
   self.encoder2 = Encoder2(self.latents[1])
    # set up the decoder submodule as well
   self.decoder2 = Decoder2(self.no_out[1])
   self.mat = OrthogMat(self.latents[0])

 #the method which will be called using VAE(parameters)
 def __call__(self, x1, x2, z_rng):
   #using the model we've set up we can input data x to the encoder
   mean1, logvar1 = self.encoder1(x1)
   mean2, logvar2 = self.encoder2(x2)
  #  mean2, logvar2 = self.encoder2(x)
   rngs = random.split(z_rng, 2)
   z1 = reparameterize(rngs[0], mean1, logvar1)
   z2 = reparameterize(rngs[1], mean2, logvar2)
   logits2 = self.decoder2(z2)
   logits1 = self.decoder1(z1)
   mat = self.mat(self.alpha*jnp.eye(self.latents[0]))
  #  U, S, V = self.mat(jnp.eye(self.latents[0]))

  #  mat = self.mat(jnp.eye(self.latents[0]))
   # return the logits as well as the mean and logvar of all
   # the latent variables distributions
  #  return logits1, mean1, logvar1, z1, logits2, mean2, logvar2, z2, U, S, V
   return logits1, mean1, logvar1, z1, logits2, mean2, logvar2, z2, mat
 

 # a function which takes a latent variable and decodes it
 # applying the sigmoid function to the output ?
 def generate(self, z_1, z_2):
    return self.decoder1(z_1), self.decoder2(z_2)
 
 def cross_gen(self, z_1, z_2, z_rng):
    rngs = random.split(z_rng, 2)
    #z_1 conditiional on z_2
    z1_cond = cond_mean(z_1, z_2, rngs[0])
    z2_cond = cond_mean(z_2, z_1, rngs[1])
    recon_x1 = self.decoder1(z1_cond)
    recon_x2 = self.decoder2(z2_cond)
    return z1_cond, recon_x1, z2_cond, recon_x2

#  def gen_from_z(self, vec):
#     print(vec.shape)
#     return nn.sigmoid(self.decoder(vec))

 # to set up a model, we automatically call vae with latents specified?
def model(latents, no_out, alpha):
  return VAE(latents, no_out, alpha)

def cond_mean(z2, z1, rng):
    #z_2 conditional on z_1
    #means from each model
    no_z1 = z1.shape[1]
    no_z2 = z2.shape[1]
    no_z = no_z1 + no_z2
    mean_vec1 = jnp.asarray([z1[:,j].mean().item() for j in range(no_z1)])
    mean_vec2 = jnp.asarray([z2[:,j].mean().item() for j in range(no_z2)])
    cov = other.est_cov(z1, z2, no_z)
    #subcovariance matrices
    cov_11 = cov[0:no_z1,0:no_z1]
    cov_12 = cov[0:no_z1,no_z1:no_z]
    cov_22 = cov[no_z1:no_z, no_z1:no_z]
    cov_term = jnp.matmul(cov_12.transpose(),jnp.linalg.inv(cov_11))
    #latent variables from model 1
    obs_z = z1
    #mean of conditional dist
    mean_cond = mean_vec2 + jnp.matmul(cov_12.transpose(),jnp.matmul(jnp.linalg.inv(cov_11),(obs_z- mean_vec1).transpose())).transpose()
    #covariance mat of conditonal dist
    # cov_mat = cov_22-jnp.matmul(cov_term, cov_12)
    # # rng = random.key(60)
    # #generate latent variables from the conditional dist
    # cond_z =  random.multivariate_normal(rng,mean=mean_cond, cov=cov_mat)
    # cond_z =  random.multivariate_normal(rng,mean=mean_cond, cov=jnp.eye(no_z2))
    # return cond_z
    return mean_cond
