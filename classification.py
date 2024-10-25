#classification file
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import time
import numpy as np
import other 
import metrics as met

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)
  
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)

# Let's upgrade it to handle batches using `vmap`

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)
  
def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)

@jit
def update(params, x, y, step_size):
  grads = grad(loss)(params, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

def class_results(train_ds, train_labs, test_ds, test_labs,
                   full_train, full_train_labs, params, num_epochs=30, batch_num=32, n_targets=10, step_size=0.01, start_range=0, end_range=392, cut=0.2):
  
  train_accuracies = []
  test_accuracies = []
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(len((full_train))//batch_num):
        x = next(train_ds)[:,start_range:end_range]
        # x = other.binarize_array(x, cut)
        y = next(train_labs)
        y = one_hot(y, n_targets)
        params = update(params, x, y, step_size)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, full_train[:,start_range:end_range], one_hot(np.array(full_train_labs), n_targets))
    test_acc = accuracy(params, test_ds[:,start_range:end_range], one_hot(test_labs[0], n_targets))
    
    #train_acc = accuracy(params, other.binarize_array(full_train[:,start_range:end_range], cut), one_hot(np.array(full_train_labs), n_targets))
    #test_acc = accuracy(params, other.binarize_array(test_ds[:,start_range:end_range], cut), one_hot(test_labs[0], n_targets))
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    # if epoch % 10 == 0:
    #   print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    #   print("Training set accuracy {}".format(train_acc))
    #   print("Test set accuracy {}".format(test_acc))
  
  return train_accuracies, test_accuracies
  # print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  # print("Training set accuracy {}".format(train_acc))
  # print("Test set accuracy {}".format(test_acc))
def class_results_all(train_ds, train_labs, test_ds, recon_test_ds, test_labs,
                   full_train, full_train_labs, params, num_epochs=30, batch_num=32, n_targets=10, step_size=0.01, start_range=0, end_range=392):
  
  train_accuracies = []
  test_accuracies = []
  recon_accuracies = []
  for epoch in range(num_epochs):
    for _ in range(len((full_train))//batch_num):
        x = next(train_ds)[:,start_range:end_range]
        y = next(train_labs)
        y = one_hot(y, n_targets)
        params = update(params, x, y, step_size)

    train_acc = accuracy(params, full_train[:,start_range:end_range], one_hot(np.array(full_train_labs), n_targets))
    test_acc = accuracy(params, test_ds[:,start_range:end_range], one_hot(test_labs[0], n_targets))
    recon_test_acc = accuracy(params, recon_test_ds[:,start_range:end_range], one_hot(test_labs[0], n_targets))
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    recon_accuracies.append(recon_test_acc)
    # if epoch % 10 == 0:
    #   print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    #   print("Training set accuracy {}".format(train_acc))
    #   print("Test set accuracy {}".format(test_acc))
  
  return train_accuracies, test_accuracies, recon_accuracies

def class_results_combine(train_ds1, train_ds2, train_labs, test_ds1, test_ds2, recon_test1, recon_test2, test_labs,
                   full_train1, full_train2, full_train_labs, params, num_epochs=30, batch_num=32, n_targets=10, step_size=0.01, start_range=0, end_range=784):
  
  train_accuracies = []
  test_accuracies = []
  recon_accuracies1 = []
  recon_accuracies2 = []
  recon_accuracies_both = []

  train_ds = jnp.concatenate([train_ds1, train_ds2], axis=1)
  test_ds = jnp.concatenate([test_ds1, test_ds2], axis=1)
  full_train = jnp.concatenate([full_train1, full_train2], axis=1)
  recon_test_ds1 = jnp.concatenate([test_ds1, recon_test2], axis=1)
  recon_test_ds2 = jnp.concatenate([recon_test1, test_ds2], axis=1)
  recon_test_ds_both = jnp.concatenate([recon_test1, recon_test2], axis=1)

  train_ds = batch_generator(train_ds, batch_num)
  train_labs = batch_generator(train_labs, batch_num)
  # test_ds = batch_generator(test_ds, batch_num)


  for epoch in range(num_epochs):
    for _ in range(len((full_train))//batch_num):
        x = next(train_ds)[:,start_range:end_range]
        y = next(train_labs)
        y = one_hot(y, n_targets)
        params = update(params, x, y, step_size)

    train_acc = accuracy(params, full_train[:,start_range:end_range], one_hot(np.array(full_train_labs), n_targets))
    test_acc = accuracy(params, test_ds[:,start_range:end_range], one_hot(test_labs[0], n_targets))
    recon_test_acc1 = accuracy(params, recon_test_ds1[:,start_range:end_range], one_hot(test_labs[0], n_targets))
    recon_test_acc2 = accuracy(params, recon_test_ds2[:,start_range:end_range], one_hot(test_labs[0], n_targets))
    recon_test_acc_both = accuracy(params, recon_test_ds_both[:,start_range:end_range], one_hot(test_labs[0], n_targets))
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    recon_accuracies1.append(recon_test_acc1)
    recon_accuracies2.append(recon_test_acc2)
    recon_accuracies_both.append(recon_test_acc_both)

    # if epoch % 10 == 0:
    #   print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    #   print("Training set accuracy {}".format(train_acc))
    #   print("Test set accuracy {}".format(test_acc))
  
  return train_accuracies, test_accuracies, recon_accuracies1, recon_accuracies2, recon_accuracies_both

def batch_generator(jax_array, batch_size):
    n = jax_array.shape[0]
    start = 0
    while True:
        if start + batch_size <= n:
            yield jax_array[start:start + batch_size]
            start += batch_size
        else:
            yield jnp.concatenate((jax_array[start:], jax_array[:(start + batch_size) % n]))
            start = (start + batch_size) % n
