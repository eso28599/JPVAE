import jax
import jax.numpy as jnp
import vae_orthog as vae
# import vae_evals as vae
import metrics as met

# defines training step functions for different constraints and data
# binary data uses cross entropy loss and continuous uses mean squared erro

# train_step_bin: binary data with orthogonal constraint implemented with cayley param
# train_step_bin_pen: binary data with orthogonal constraint implemented with a penalty
# train_step: cont. data with orthogonal constraint implemented with cayley param
# train_step_pen: cont. data with orthogonal constraint implemented with a penalty

def orthog_pen(C, alpha):
  return jnp.linalg.norm(jnp.matmul(C, C.transpose()) - jnp.square(alpha) * jnp.eye(C.shape[0]))  + jnp.linalg.norm(jnp.matmul(C.transpose(), C) - jnp.square(alpha)*jnp.eye(C.shape[1]))

# training step for binary data with orthogonal constraints implemented with cayley param
def ts_bin_orthog(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero=1):
  #define the loss function
  @jax.jit
  def loss_fn(params):
   logits1, mean1, logvar1, _, logits2, mean2, logvar2, _, C_mat = vae.model(latents, num_out, alpha).apply({'params': params}, batch1, batch2, z_rng)
    # calculate the entropy loss between the reconstructed x and the original (batch)
   bce_loss1 = jnp.mean(met.bce_with_logits(logits1, batch1))
   bce_loss2 = jnp.mean(met.bce_with_logits(logits2, batch2))
   kld_loss = jnp.mean(met.kl_div(mean1,logvar1, mean2, logvar2, C_mat, latents))
   loss = bce_loss1 + bce_loss2 + beta * kld_loss
   
   return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

train_step_bin = jax.jit(ts_bin_orthog, static_argnums=(4, 5, 6, 8))

# training step for binary data with orthogonal constraints implemented with penalisation
def ts_bin_orthog_pen(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero=1, lambda_val = 20):
  #define the loss function
  @jax.jit
  def loss_fn(params):
   logits1, mean1, logvar1, _, logits2, mean2, logvar2, _, C_mat = vae.model(latents, num_out, alpha).apply({'params': params}, batch1, batch2, z_rng)
    # calculate the entropy loss between the reconstructed x and the original (batch)
   bce_loss1 = jnp.mean(met.bce_with_logits(logits1, batch1))
   bce_loss2 = jnp.mean(met.bce_with_logits(logits2, batch2))
   kld_loss = jnp.mean(met.kl_div(mean1,logvar1, mean2, logvar2, C_mat, latents))
#    kld_loss = jnp.mean(met.kl_div_noC(mean1,logvar1, mean2, logvar2))
   orthog_term = orthog_pen(C_mat, alpha)
   loss = bce_loss1 + bce_loss2 + beta * kld_loss +  lambda_val * orthog_term
   
   return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

train_step_bin_pen = jax.jit(ts_bin_orthog_pen, static_argnums=(4, 5, 6, 8))


def ts_orthog(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero=1):
  #define the loss function
  @jax.jit
  def loss_fn(params):
   logits1, mean1, logvar1, _, logits2, mean2, logvar2, _, C_mat = vae.model(latents, num_out, alpha).apply({'params': params}, batch1, batch2, z_rng)
    # calculate the entropy loss between the reconstructed x and the original (batch)
   bce_loss1 = jnp.mean(met.mse_loss(logits1, batch1))
   bce_loss2 = jnp.mean(met.mse_loss(logits2, batch2))
   kld_loss = jnp.mean(met.kl_div(mean1,logvar1, mean2, logvar2, zero*C_mat, latents))
   loss = bce_loss1 + bce_loss2 + beta * kld_loss
   
   return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

train_step = jax.jit(ts_orthog, static_argnums=(4, 5, 6, 8))

def ts_orthog_pen(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero=1, lambda_val = 20):
  #define the loss function
  @jax.jit
  def loss_fn(params):
   logits1, mean1, logvar1, _, logits2, mean2, logvar2, _, C_mat = vae.model(latents, num_out, alpha).apply({'params': params}, batch1, batch2, z_rng)
    # calculate the entropy loss between the reconstructed x and the original (batch)
   bce_loss1 = jnp.mean(met.mse_loss(logits1, batch1))
   bce_loss2 = jnp.mean(met.mse_loss(logits2, batch2))
   kld_loss = jnp.mean(met.kl_div(mean1,logvar1, mean2, logvar2, zero*C_mat, latents))
   orthog_term = orthog_pen(C_mat, alpha)
   loss = bce_loss1 + bce_loss2 + beta * kld_loss + lambda_val * orthog_term 
   
   return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

train_step_pen = jax.jit(ts_orthog_pen, static_argnums=(4, 5, 6, 8))


def ts_noC(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero=1):
  #define the loss function
  @jax.jit
  def loss_fn(params):
   logits1, mean1, logvar1, _, logits2, mean2, logvar2, _, _ = vae.model(latents, num_out, alpha).apply({'params': params}, batch1, batch2, z_rng)
    # calculate the entropy loss between the reconstructed x and the original (batch)
   bce_loss1 = jnp.mean(met.bce_with_logits(logits1, batch1))
   bce_loss2 = jnp.mean(met.bce_with_logits(logits2, batch2))
   kld_loss = jnp.mean(met.kl_div_noC(mean1,logvar1, mean2, logvar2))
   loss = bce_loss1 + bce_loss2 + beta * kld_loss
   
   return loss
  grads = jax.grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads)

train_step_noC = jax.jit(ts_noC, static_argnums=(4, 5, 6, 8))

    