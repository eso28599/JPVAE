import jax
import jax.numpy as jnp
import vae_orthog as vae
import metrics as met
from jax import random
from flax import linen as nn
import other
from flax.training import train_state
import optax
import utils as vae_utils
import ml_collections
import training_steps as ts 

def eval(params, images1, images2, z_1, z_2, z_rng, latents, num_out, alpha, binary,param, zero=False, print_images=False, h=14, w=28):
  rngs = random.split(z_rng, 50)
  n_images = images1.shape[0]
  def eval_model(vae):
    loss = []
    for i in range(50):
      recon_images1, mean1, logvar1, z1, recon_images2, mean2, logvar2, z2, mat = vae(images1, images2, rngs[i])
      matrices = other.make_matrices(latents, mat)
      if zero:
        metrics = met.compute_metrics_zero(recon_images1, recon_images2, images1, images2,
                                  mean1, logvar1, mean2, logvar2, False)
      elif binary:
        metrics = met.compute_metrics_bin(recon_images1, recon_images2, images1, images2,
                                  mean1, logvar1, mean2, logvar2, matrices, False)
      else:
        metrics = met.compute_metrics(recon_images1, recon_images2, images1, images2,
                                  mean1, logvar1, mean2, logvar2, matrices, False)
      loss.append(metrics['loss'])

    ll = jnp.mean(nn.activation.logsumexp(-jnp.asarray(loss),axis=0) - jnp.ones(n_images)*jnp.log(n_images/20))
    recon_images1, mean1, logvar1, z1, recon_images2, mean2, logvar2, z2, mat = vae(images1, images2, rngs[0])
    if zero:
      metrics = met.compute_metrics_zero(recon_images1, recon_images2, images1, images2, mean1, logvar1, mean2, logvar2, other.make_matrices(latents, mat))
    elif binary:
      metrics = met.compute_metrics_bin(recon_images1, recon_images2, images1, images2, mean1, logvar1, mean2, logvar2, other.make_matrices(latents, mat))
    else:
      metrics = met.compute_metrics(recon_images1, recon_images2, images1, images2, mean1, logvar1, mean2, logvar2, other.make_matrices(latents, mat))
    if print_images:
      comparison = jnp.concatenate([
          images1[:8].reshape(-1, h, w, 1),
          recon_images1[:8].reshape(-1, h, w, 1),
          images2[:8].reshape(-1, h, w, 1),
          recon_images2[:8].reshape(-1, h, w, 1)
      ])
      generate_images1, generate_images2 = vae.generate(z_1, z_2)
      generate_images1 = generate_images1.reshape(-1, h, w, 1)
      generate_images2 = generate_images2.reshape(-1, h, w, 1)
    else:
      comparison = jnp.zeros((1,1))
      generate_images1 = jnp.zeros((1,1))
      generate_images2 = jnp.zeros((1,1))
    
    return metrics, comparison, generate_images1, generate_images2, ll, z1, z2, mat
  return nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
eval_f = jax.jit(eval, static_argnums=(6,7, 8, 9, 10,11,12))


def get_z(params, images1, images2, z_rng, latents,num_out, alpha):
  def eval_model(vae):
    recon_images1, mean1, logvar1, z1, recon_images2, mean2, logvar2, z2, _= vae(images1, images2, z_rng)
    return z1, recon_images1, z2, recon_images2, mean1, mean2, logvar1, logvar2
  # zs1, recons1, zs2, recons2, mean1, = nn.apply(eval_model, vae.model(latents))({'params': params})
  zs1, recons1, zs2, recons2, mean1, mean2, logvar1, logvar2, _ = nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
  # return {'z_v1':zs1, 'z_v2':zs2, 'rec_v1':recons1, 'rec_v2':recons2}
  return {'z_v1':zs1, 'z_v2':zs2, 'rec_v1':recons1, 'rec_v2':recons2, 'mean1':mean1, 'mean2':mean2, 'logvar1':logvar1, 'logvar2':logvar2}

def get_stats(params, images1, images2, z_rng, latents, num_out, alpha):
  def eval_model(vae):
    _, mean1, logvar1, _, _, mean2, logvar2, _, mat= vae(images1, images2, z_rng)
    return mean1, mean2, logvar1, logvar2, mat
  mean1, mean2, logvar1, logvar2, mat  = nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
  return mean1, mean2, logvar1, logvar2, mat

def get_all_z(params, images1, images2, z_rng, latents, num_out, alpha, binary):
  def eval_model(vae):
    recon_images1, mean1, _, z1, recon_images2, mean2, _, z2, _= vae(images1, images2, z_rng)
    return z1, recon_images1, z2, recon_images2, mean1, mean2
    # recon_images1, mean1, _, _, recon_images2, mean2, _, _= vae(images1, images2, z_rng)
    # return mean1, recon_images1, mean2, recon_images2

  zs1, recons1, zs2, recons2, mean1, mean2 = nn.apply(eval_model, vae.model(latents, num_out, alpha))({'params': params})
  def apply_dec(vae):
    z1_cond, recon_x1, z2_cond, recon_x2 = vae.cross_gen(zs1, zs2, z_rng)
    return z1_cond, recon_x1, z2_cond, recon_x2

  z1_cond, recon_x1, z2_cond, recon_x2 = nn.apply(apply_dec, vae.model(latents, num_out, alpha))({'params': params})
  full_res = {'z_v1':zs1, 'z_v2':zs2, 'rec_v1':recons1, 'rec_v2':recons2, 'mean1':mean1, 'mean2':mean2}
  cond_res = {'z1_cond':z1_cond, 'recon_x1':recon_x1, 'z2_cond':z2_cond, 'recon_x2':recon_x2}

  #calculate losses
  if binary:
    loss1 = jnp.mean(met.bce_with_logits(recons1, images1))
    loss2 = jnp.mean(met.bce_with_logits(recons2, images2))
    loss_z1 = jnp.mean(met.bce_with_logits(recon_x1, images1))
    loss_z2 = jnp.mean(met.bce_with_logits(recon_x2, images2))
  else:
    loss1 = jnp.mean(met.mse_loss(recons1, images1))
    loss2 = jnp.mean(met.mse_loss(recons2, images2))
    loss_z1 = jnp.mean(met.mse_loss(recon_x1, images1))
    loss_z2 = jnp.mean(met.mse_loss(recon_x2, images2))
  # losses = {'loss1':loss1, 'loss2':loss2, 'loss_z1':loss_z1, 'loss_z2':loss_z2}
  losses = jnp.array([loss1, loss2, loss_z1, loss_z2])
  return full_res, cond_res, losses



def train_and_eval_split(config: ml_collections.ConfigDict, train_ds, test_ds, full_train,
                         num_examples, key_val, zero = False, binary=True, orthog = True, param = True, 
                         cut = 0.02, print_images = False, toy = False, lambda_val =20):
  """Train and evaulate pipeline."""
  rng = random.key(key_val)
  rng, key = random.split(rng)

  init_data = jnp.ones((config.batch_size, config.num_out[0]), jnp.float32)
  params = vae.model(config.latents, config.num_out, config.alpha).init(key, init_data, init_data, rng)['params']

  state = train_state.TrainState.create(
      apply_fn=vae.model(config.latents, config.num_out, config.alpha).apply,
      params=params,
      tx=optax.adam(config.learning_rate),
  )

  rng, z_key, eval_rng = random.split(rng, 3)
  steps_per_epoch = (
      num_examples // config.batch_size
    )

  if zero:
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero,lambda_va):
        return ts.train_step_noC(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero)
  elif (not binary) & orthog & param: #cont., orthog by param
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero,lambda_va):
        return ts.train_step(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero)
  elif (not binary) & orthog & (not param): 
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero, lambda_val):
        return ts.train_step_pen(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero, lambda_val)
  elif binary & orthog & param: #binary, orthog by param
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero, lambda_val):
        return ts.train_step_bin(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero)
  else: #binary, orthog with a penalty
      def train_step(state, batch1, batch2, z_rng, latents, num_out, alpha, beta, zero, lambda_val):
        return ts.train_step_bin_pen(state, batch1, batch2, z_rng,latents, num_out, alpha, beta, zero, lambda_val)
    
  
  beta_vec = other.make_beta(steps_per_epoch,1, 1)
  if toy:
    test_ds1 = jnp.expand_dims(jnp.array(test_ds[:,0]),axis=1)
    test_ds2 = jnp.expand_dims(jnp.array(test_ds[:,1]),axis=1)
    full_train1 = jnp.expand_dims(jnp.array(full_train[:, 0]), axis=1)
    full_train2 = jnp.expand_dims(jnp.array(full_train[:, 1]), axis=1)
    z_1 = 1
    z_2 = 1
  else:
    test_ds_bin = other.binarize_array(test_ds, cut)
    test_ds1 = test_ds_bin[:,0:392]
    test_ds2 = test_ds_bin[:, 392:784]
    full_train_bin = other.binarize_array(full_train, cut)
    full_train1 = full_train_bin[:,0:392]
    full_train2 = full_train_bin[:, 392:784]
    z_1 = random.normal(z_key, (64, config.latents[0]))
    z_2 = random.normal(z_key, (64, config.latents[1]))

  for epoch in range(config.num_epochs):
    for i in range(steps_per_epoch):
      batch = next(train_ds)
      if toy:
        batch1 = jnp.expand_dims(jnp.array(batch[:,0]),axis=1)
        batch2 = jnp.expand_dims(jnp.array(batch[:, 1]), axis=1)
      else:
        batch_bin = other.binarize_array(batch,cut)
        batch1 = batch_bin[:,0:392]
        batch2 = batch_bin[:, 392:784]
      rng, key = random.split(rng)
      beta = float(beta_vec[i])
      state = train_step(state, batch1, batch2, key, config.latents, config.num_out, config.alpha,
                                beta,zero,lambda_val)
      # return get_stats(state.params, full_train1, full_train2, eval_rng, config.latents, config.num_out, config.alpha)
      # return get_all_z(state.params, full_train1, full_train2, eval_rng, config.latents, config.num_out, config.alpha, binary)
    metrics, comparison, sample1, sample2, ll, _, _, mat= eval_f(
        state.params, test_ds1, test_ds2, z_1, z_2, eval_rng, config.latents, config.num_out, config.alpha,  binary, param, zero, print_images
    )
    if(print_images):
      vae_utils.save_image(
          comparison, f'{config.results_path}/reconstruction_{epoch}.png', nrow=8
      )
      vae_utils.save_image(sample1, f'{config.results_path}/sample1_{epoch}.png', nrow=8)
      vae_utils.save_image(sample2, f'{config.results_path}/sample2_{epoch}.png', nrow=8)
    if (epoch + 1) % 5 == 0:
      print(
          'eval epoch: {}, loss: {:.4f}, BCE: {:.4f}, KLD: {:.4f}, LL: {:.4f}'.format(
              epoch + 1, metrics['loss'], metrics['bce'], metrics['kld'], ll)
      )

  rng = random.key(key_val)
  full_res, full_cond, full_loss = get_all_z(state.params, full_train1,
                                            full_train2, rng, config.latents, config.num_out, config.alpha, binary)
  test_res, test_cond, test_loss = get_all_z(state.params, test_ds1, 
                                                             test_ds2, rng, config.latents, config.num_out, config.alpha, binary)

  return full_res, state.params,  test_res, full_cond, test_cond, full_loss, test_loss, mat
