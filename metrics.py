import jax
import jax.numpy as jnp
from flax import linen as nn

# file containing metrics for elbo calculation 
# for both binary data - binary cross entropy (bce_with logits)
# and continuous data - mean squared error (mse_loss)

@jax.vmap
def bce_with_logits2(logits, labels):
  logits1 = nn.log_sigmoid(logits)
  return -jnp.sum(labels * logits1 + (1. - labels) * jnp.log(-jnp.expm1(logits1)))

bce_with_logits = jax.jit(bce_with_logits2)

@jax.vmap
def mse_loss2(original, reconstructed):
    return jnp.sum(jnp.square(original - reconstructed))

mse_loss = jax.jit(mse_loss2)

def kl(mean1, logvar1, mean2, logvar2, matrices):
  D1 = matrices['D1']
  D2 = matrices['D2']
  D1CT = matrices['D1CT']
  D2C = matrices['D2C']
  term1 = -jnp.sum(2 + logvar1) -jnp.sum(logvar2)
  means =  jnp.matmul(mean1,jnp.matmul(D1, mean1)) + jnp.matmul(mean2,jnp.matmul(D2, mean2))
  vars = jnp.trace(jnp.matmul(D1, jnp.diag(jnp.exp(logvar1)))) + jnp.trace(jnp.matmul(D2, jnp.diag(jnp.exp(logvar2))))
  term2 = -(matrices['log_detD'] + jnp.matmul(mean1,jnp.matmul(D1CT,mean2))+ jnp.matmul(mean2, jnp.matmul(D2C,mean1)))
  return 0.5*(means + term1 + vars + term2)

kl_div1 = jax.vmap(kl, in_axes=(0,0,0,0, None))
kl_div_og = jax.jit(kl_div1)


def kl_noC(mean1, logvar1, mean2, logvar2):
  term1 = -jnp.sum(2 + logvar1) -jnp.sum(logvar2)
  means =  jnp.matmul(mean1,mean1) + jnp.matmul(mean2,mean2)
  vars = jnp.sum(jnp.exp(logvar1)) + jnp.sum(jnp.exp(logvar2))
  term2 = -(jnp.log(mean1.shape[0]))
  return 0.5*(means + term1 + vars + term2)

kl_div_noC1 = jax.vmap(kl_noC, in_axes=(0,0,0,0))
kl_div_noC = jax.jit(kl_div_noC1)

def kl(mean1, logvar1, mean2, logvar2, C, no_latents):

  D1 = jnp.linalg.inv(jnp.eye(no_latents[0]) - jnp.matmul(jnp.transpose(C), C))
  D2 = jnp.linalg.inv(jnp.eye(no_latents[1]) - jnp.matmul(C, jnp.transpose(C)))
  D1CT = jnp.matmul(D1,jnp.transpose(C))
  D2C = jnp.matmul(D2, C)
  term1 = -jnp.sum(2 + logvar1) -jnp.sum(logvar2)
  means =  jnp.matmul(mean1,jnp.matmul(D1, mean1)) + jnp.matmul(mean2,jnp.matmul(D2, mean2))
  vars = jnp.trace(jnp.matmul(D1, jnp.diag(jnp.exp(logvar1)))) + jnp.trace(jnp.matmul(D2, jnp.diag(jnp.exp(logvar2))))
  term2 = -(jnp.log(jnp.linalg.det(D1)) + jnp.matmul(mean1,jnp.matmul(D1CT,mean2))+ jnp.matmul(mean2, jnp.matmul(D2C,mean1)))
  return 0.5*(means + term1 + vars + term2)

kl_div1 = jax.vmap(kl, in_axes=(0,0,0,0, None, None))
kl_div = jax.jit(kl_div1, static_argnums=(5))

def compute_metrics_bin(recon_x1, recon_x2, batch1, batch2, mean1, logvar1, mean2, logvar2, matrices, average=True):
  bce_loss1 = (bce_with_logits(recon_x1, batch1))
  bce_loss2 = (bce_with_logits(recon_x2, batch2))
  kld_loss = (kl_div_og(mean1, logvar1, mean2, logvar2, matrices))
  bce_loss = bce_loss1 + bce_loss2
  loss = bce_loss + kld_loss
  if average:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': jnp.mean(loss)}
  else:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': (loss)}


def compute_metrics(recon_x1, recon_x2, batch1, batch2, mean1, logvar1, mean2, logvar2, matrices, average=True):
  mse_loss1 = (mse_loss(recon_x1, batch1))
  mse_loss2 = (mse_loss(recon_x2, batch2))
  kld_loss = (kl_div_og(mean1, logvar1, mean2, logvar2, matrices))
  bce_loss = mse_loss1 + mse_loss2
  loss = bce_loss + kld_loss
  if average:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': jnp.mean(loss)}
  else:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': (loss)}
  
def compute_metrics_zero(recon_x1, recon_x2, batch1, batch2, mean1, logvar1, mean2, logvar2, average=True):
  mse_loss1 = (bce_with_logits(recon_x1, batch1))
  mse_loss2 = (bce_with_logits(recon_x2, batch2))
  kld_loss = kl_div_noC(mean1, logvar1, mean2, logvar2)
  bce_loss = mse_loss1 + mse_loss2
  loss = bce_loss + kld_loss
  if average:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': jnp.mean(loss)}
  else:
    return {'bce': jnp.mean(bce_loss), 'kld': jnp.mean(kld_loss), 'loss': (loss)}