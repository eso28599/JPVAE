import ml_collections
import jax.numpy as jnp

def get_config(num_latents1 = 20, num_latents2 = 20, num_out1= 392, 
               num_out2=392, alpha= 0.9, batch_num = 32, number_epochs = 30, learning_rate = 1e-3):
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.alpha = alpha
  config.learning_rate = learning_rate
  config.latents = (num_latents1, num_latents2)
  config.batch_size = batch_num
  config.num_epochs = number_epochs
  config.num_out = (num_out1, num_out2)
  return config

def make_beta(steps, num_epochs, M):
  total_steps = steps*num_epochs
  half_step = total_steps//(2*M)
  cycle = jnp.concatenate((jnp.linspace(0,1, half_step), jnp.ones(half_step)))
  # cycle = jnp.linspace(0,1, 2*half_step)
  end_cycle = jnp.ones(total_steps - half_step*2*(M-1))
  return jnp.concatenate((end_cycle, jnp.tile(cycle,M-1)))

def make_matrices(no_latents, matrix):
  D_1 = jnp.linalg.inv(jnp.eye(no_latents[0]) - jnp.matmul(jnp.transpose(matrix), matrix))
  D_2 = jnp.linalg.inv(jnp.eye(no_latents[1]) - jnp.matmul(matrix, jnp.transpose(matrix)))

  return {'C':matrix, 'D1':D_1, 'D2':D_2,'D1CT':jnp.matmul(D_1,jnp.transpose(matrix)),
                    'D2C':jnp.matmul(D_2, matrix),'log_detD':jnp.log(jnp.linalg.det(D_1))}

def is_pos_def(C):
    cov = jnp.block([
        [jnp.eye(C.shape[1]),            jnp.transpose(C)],
        [C, jnp.eye(C.shape[0])]
        ])
    return jnp.all(jnp.linalg.eigh(cov).eigenvalues > 0)

def est_cov(z1, z2, no_z):
    all_zs = jnp.concatenate((z1, z2),axis=1)#latent variables stacked
    mean_z = jnp.asarray([all_zs[:,j].mean().item() for j in range(no_z)]) #mean of each latent variable
    norm = all_zs - mean_z
    cov = jnp.matmul(norm.transpose(),norm)/(z1.shape[0])
    return cov

def binarize_array(arr, cut):
    binarized_arr = jnp.where(arr > cut, 1, 0)
    return binarized_arr

def prepare_image(x, batch_size):
  x1 = jnp.reshape(jnp.squeeze(jnp.array(x[0])), (batch_size,784))
  x2 = jnp.reshape(jnp.squeeze(jnp.array(x[1])), (batch_size,784))
  return x1, x2
   
def save_as_matrix(matrix, name, vals):
    matrix = jnp.array(matrix)
    matrix = jnp.concatenate((vals, matrix), axis=1)
    jnp.save(name, matrix)