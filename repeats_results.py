import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# export TF2_BEHAVIOR=1
# export TPU_ML_PLATFORM="Tensorflow"
# export TF_CPP_MIN_LOG_LEVEL=1
# load packages
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import jax
import jax.numpy as jnp
from jax import random
import ml_collections
from ml_collections import config_flags
import tensorflow_datasets as tfds
from absl import logging
import input_pipeline
import matplotlib.pyplot as plt
# import seaborn as sb
import other
import time
# from scipy.stats import normaltest
import classification as cla
# import torch
from importlib import reload
import vae_orthog as vae
import train_vae_withCorthog as train
import utils
import numpy as np
import train_vae_withCevals as train_evals
import pandas as pd

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["TPU_ML_PLATFORM"] = "TPU"


no_latents1 = 20
no_latents2 = 20

# choose one configuration and set results path
# results_path = 'results_9'
# c_flags1 = other.get_config(no_latents1, no_latents2, alpha = 0.9,       
#                             number_epochs=30, results_path=results_path)
# results_path = 'results_1'
# c_flags1 = other.get_config(no_latents1, no_latents2, alpha=0.1,       
#                             number_epochs=30, results_path=results_path)
results_path = 'results_5'
c_flags1 = other.get_config(no_latents1, no_latents2, alpha=0.5,       
                            number_epochs=30, results_path=results_path)
ds_builder = tfds.builder('mnist')
ds_builder.download_and_prepare()



layer_sizes = [392, 512, 512, 10]
layer_sizes_full = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 30
batch_size = 32
n_targets = 10
params = cla.init_network_params(layer_sizes, random.key(0))



# save train and test accuracy in a csv file
import csv
results = np.zeros((5,6))

# run the model 5 times on the original data
for i in range(5):
    logging.info('Initializing dataset.')
    train_ds1, train_labs1 = input_pipeline.build_train_set(c_flags1.batch_size, ds_builder,40+i)
    test_ds1, test_labs1 = input_pipeline.build_test_set(ds_builder)
    full_train_ds1, full_train_labs1 = input_pipeline.build_full_train(ds_builder)
    logging.info('Initializing model.')
    params = cla.init_network_params(layer_sizes, random.key(i))
    params_full = cla.init_network_params(layer_sizes_full, random.key(i))
    train_res, test_res = cla.class_results(train_ds1, train_labs1,
                test_ds1, test_labs1,
                    full_train_ds1, full_train_labs1, params, num_epochs=15, batch_num=batch_size, cut=0.2)
    train_res2, test_res2 = cla.class_results(train_ds1, train_labs1,
               test_ds1, test_labs1,
                   full_train_ds1, full_train_labs1, params, num_epochs=15, batch_num=batch_size, start_range=392, end_range=784)
    # classifcation based on both views
    train_res_both, test_res_both = cla.class_results(train_ds1, train_labs1,
                test_ds1, test_labs1,
                    full_train_ds1, full_train_labs1, params_full, num_epochs=15, batch_num=batch_size, cut=0.2, end_range=784)
    #print train and test accuracy
    results[i,] = (train_res[-1], test_res[-1], train_res2[-1], test_res2[-1], train_res_both[-1], test_res_both[-1])

    with open(f'{results_path}/original_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)

#C=0
results_zero = np.zeros((5,21))

for i in range(5):

    logging.info('Initializing dataset.')
    train_ds, train_labs = input_pipeline.build_train_set(c_flags1.batch_size, ds_builder,40+i)
    test_ds, test_labs = input_pipeline.build_test_set(ds_builder)
    full_train_ds, full_train_labs = input_pipeline.build_full_train(ds_builder)
    logging.info('Initializing model.')
    params = cla.init_network_params(layer_sizes, random.key(i))
    params_full = cla.init_network_params(layer_sizes_full, random.key(i))
    test_zero = train.train_and_eval_split(c_flags1, train_ds, test_ds, 
                full_train_ds, 20000, (30+i), zero=True)
    result = test_zero
    au1 = sum(jnp.diag(jnp.cov(jnp.transpose(test_zero[0]['mean1'])))>0.01)
    au2 = sum(jnp.diag(jnp.cov(jnp.transpose(test_zero[0]['mean2'])))>0.01)
    train_acc_, test_acc_, recon_acc_ = cla.class_results_all(cla.batch_generator(result[0]['rec_v1'], batch_size), cla.batch_generator(full_train_labs, batch_size),
               result[2]['rec_v1'], result[4]['recon_x1'], test_labs,
                   result[0]['rec_v1'], full_train_labs, params, num_epochs=50, batch_num=batch_size)
    result = test_zero
    train_data = result[0]['rec_v2']
    train_labs = full_train_labs
    testing_data = result[2]['rec_v2']
    testing_labs = test_labs
    recon_data = result[4]['recon_x2']
    view = 2
    train_acc_2, test_acc_2, recon_acc_2 = cla.class_results_all(cla.batch_generator(train_data, batch_size), cla.batch_generator(train_labs, batch_size),
               testing_data, recon_data, testing_labs,
                   train_data, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    # decoded data
    train_data1 = result[3]['recon_x1']
    testing_data1 = result[4]['recon_x1']
    view = 1
    train_acc, test_acc = cla.class_results(cla.batch_generator(train_data1, batch_size), cla.batch_generator(full_train_labs, batch_size),
                testing_data1, test_labs,
                    train_data1, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    train_data2 = result[3]['recon_x2']
    testing_data2 = result[4]['recon_x2']
    view = 2
    train_acc2, test_acc2 = cla.class_results(cla.batch_generator(train_data2, batch_size), cla.batch_generator(full_train_labs, batch_size),
                testing_data2, test_labs,
                    train_data2, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    # classification based on both views
    train_accuracies, test_accuracies, recon_accuracies1, recon_accuracies2, recon_accuracies_both = cla.class_results_combine(result[0]['rec_v1'],result[0]['rec_v2'], full_train_labs,
                               result[2]['rec_v1'], result[2]['rec_v2'],  result[4]['recon_x1'], result[4]['recon_x2'], test_labs,
                   result[0]['rec_v1'], result[0]['rec_v2'], full_train_labs, params_full, num_epochs=50, batch_num=batch_size, end_range=784)

    results_zero[i,] = (train_acc_[-1], test_acc_[-1], recon_acc_[-1],
                   train_acc_2[-1],  test_acc_2[-1], recon_acc_2[-1],
                   train_acc[-1], test_acc[-1],
                     train_acc2[-1], test_acc2[-1],
                     train_accuracies[-1], test_accuracies[-1], recon_accuracies1[-1], recon_accuracies2[-1], recon_accuracies_both[-1],
                     au1, au2,
                       result[6][0], result[6][1], result[6][2], result[6][3])
    
    with open(f'{results_path}/zero_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results_zero)



#C is orthogonal
results_orthog = np.zeros((5,21))

for i in range(5):
    logging.info('Initializing dataset.')
    train_ds, train_labs = input_pipeline.build_train_set(c_flags1.batch_size, ds_builder,40+i)
    test_ds, test_labs = input_pipeline.build_test_set(ds_builder)
    full_train_ds, full_train_labs = input_pipeline.build_full_train(ds_builder)
    logging.info('Initializing model.')
    params = cla.init_network_params(layer_sizes, random.key(i))
    params_full = cla.init_network_params(layer_sizes_full, random.key(i))
    test_orthog_param = train.train_and_eval_split(c_flags1, train_ds, test_ds, full_train_ds, 20000, (30+i))
    result = test_orthog_param
    au1 = sum(jnp.diag(jnp.cov(jnp.transpose(test_orthog_param[0]['mean1'])))>0.01)
    au2 = sum(jnp.diag(jnp.cov(jnp.transpose(test_orthog_param[0]['mean2'])))>0.01)
    train_acc_, test_acc_, recon_acc_ = cla.class_results_all(cla.batch_generator(result[0]['rec_v1'], batch_size),
                 cla.batch_generator(full_train_labs, batch_size),
               result[2]['rec_v1'], result[4]['recon_x1'], test_labs,
                   result[0]['rec_v1'], full_train_labs, params, num_epochs=50, batch_num=batch_size)
    train_data = result[0]['rec_v2']
    train_labs = full_train_labs
    testing_data = result[2]['rec_v2']
    testing_labs = test_labs
    recon_data = result[4]['recon_x2']
    view = 2
    train_acc_2, test_acc_2, recon_acc_2 = cla.class_results_all(cla.batch_generator(train_data, batch_size), cla.batch_generator(train_labs, batch_size),
               testing_data, recon_data, testing_labs,
                   train_data, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    # decoded data
    train_data = result[3]['recon_x1']
    testing_data = result[4]['recon_x1']
    view = 1
    train_acc, test_acc = cla.class_results(cla.batch_generator(train_data, batch_size), cla.batch_generator(full_train_labs, batch_size),
                testing_data, test_labs,
                    train_data, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    train_data = result[3]['recon_x2']
    testing_data = result[4]['recon_x2']
    view = 2
    train_acc2, test_acc2 = cla.class_results(cla.batch_generator(train_data, batch_size), cla.batch_generator(full_train_labs, batch_size),
                testing_data, test_labs,
                    train_data, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    # classification based on both views
    train_accuracies, test_accuracies, recon_accuracies1, recon_accuracies2, recon_accuracies_both = cla.class_results_combine(result[0]['rec_v1'],result[0]['rec_v2'], full_train_labs,
                               result[2]['rec_v1'], result[2]['rec_v2'],  result[4]['recon_x1'], result[4]['recon_x2'], test_labs,
                   result[0]['rec_v1'], result[0]['rec_v2'], full_train_labs, params_full, num_epochs=50, batch_num=batch_size, end_range=784)

    results_orthog[i,] = (train_acc_[-1], test_acc_[-1], recon_acc_[-1],
                   train_acc_2[-1],  test_acc_2[-1], recon_acc_2[-1],
                   train_acc[-1], test_acc[-1],
                     train_acc2[-1], test_acc2[-1],
                     train_accuracies[-1], test_accuracies[-1], recon_accuracies1[-1], recon_accuracies2[-1], recon_accuracies_both[-1],
                     au1, au2,
                       result[6][0], result[6][1], result[6][2], result[6][3])
    
    with open(f'{results_path}/orthog_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results_orthog)

#C is not orthogonal
results_eval = np.zeros((5,21))

for i in range(5):
    logging.info('Initializing dataset.')
    train_ds, train_labs = input_pipeline.build_train_set(c_flags1.batch_size, ds_builder,40+i)
    test_ds, test_labs = input_pipeline.build_test_set(ds_builder)
    full_train_ds, full_train_labs = input_pipeline.build_full_train(ds_builder)
    logging.info('Initializing model.')
    params = cla.init_network_params(layer_sizes, random.key(i))
    params_full = cla.init_network_params(layer_sizes_full, random.key(i))
    test_eval =  train_evals.train_and_eval_split(c_flags1, train_ds, test_ds, full_train_ds, 20000, 30+i)
    result = test_eval
    au1 = sum(jnp.diag(jnp.cov(jnp.transpose(test_eval[0]['mean1'])))>0.01)
    au2 = sum(jnp.diag(jnp.cov(jnp.transpose(test_eval[0]['mean2'])))>0.01)
    train_acc_, test_acc_, recon_acc_ = cla.class_results_all(cla.batch_generator(result[0]['rec_v1'], batch_size),
                 cla.batch_generator(full_train_labs, batch_size),
               result[2]['rec_v1'], result[4]['recon_x1'], test_labs,
                   result[0]['rec_v1'], full_train_labs, params, num_epochs=50, batch_num=batch_size)
    train_data = result[0]['rec_v2']
    train_labs = full_train_labs
    testing_data = result[2]['rec_v2']
    testing_labs = test_labs
    recon_data = result[4]['recon_x2']
    view = 2
    train_acc_2, test_acc_2, recon_acc_2 = cla.class_results_all(cla.batch_generator(train_data, batch_size), cla.batch_generator(train_labs, batch_size),
               testing_data, recon_data, testing_labs,
                   train_data, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    # decoded data
    train_data = result[3]['recon_x1']
    testing_data = result[4]['recon_x1']
    view = 1
    train_acc, test_acc = cla.class_results(cla.batch_generator(train_data, batch_size), cla.batch_generator(full_train_labs, batch_size),
                testing_data, test_labs,
                    train_data, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    train_data = result[3]['recon_x2']
    testing_data = result[4]['recon_x2']
    view = 2
    
    train_acc2, test_acc2 = cla.class_results(cla.batch_generator(train_data, batch_size), cla.batch_generator(full_train_labs, batch_size),
                testing_data, test_labs,
                    train_data, full_train_labs, params, num_epochs=50, batch_num=batch_size)
    # classification based on both views
    train_accuracies, test_accuracies, recon_accuracies1, recon_accuracies2, recon_accuracies_both = cla.class_results_combine(result[0]['rec_v1'],result[0]['rec_v2'], full_train_labs,
                               result[2]['rec_v1'], result[2]['rec_v2'],  result[4]['recon_x1'], result[4]['recon_x2'], test_labs,
                   result[0]['rec_v1'], result[0]['rec_v2'], full_train_labs, params_full, num_epochs=50, batch_num=batch_size, end_range=784)

    results_eval[i,] = (train_acc_[-1], test_acc_[-1], recon_acc_[-1],
                   train_acc_2[-1],  test_acc_2[-1], recon_acc_2[-1],
                   train_acc[-1], test_acc[-1],
                     train_acc2[-1], test_acc2[-1],
                     train_accuracies[-1], test_accuracies[-1], recon_accuracies1[-1], recon_accuracies2[-1], recon_accuracies_both[-1],
                     au1, au2,
                       result[6][0], result[6][1], result[6][2], result[6][3])
    
    with open(f'{results_path}/eval_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results_eval)
#load original results
df_orig = pd.read_csv(f'{results_path}/original_results.csv', header=None)
mean_orig = df_orig.mean(axis=0)
#pad with None for those results not applicable for original
mean_orig = pd.concat([pd.Series([mean_orig[0], mean_orig[1], None, mean_orig[2], mean_orig[3]]),
                       pd.Series([None] * 5, index=range(5,10)),
                                 pd.Series([mean_orig[4], mean_orig[5]], index=range(10,12)),
                                   pd.Series([None] * 8, index=range(13,21))])
sd_orig = df_orig.std(axis=0)
sd_orig = pd.concat([pd.Series([sd_orig[0], sd_orig[1], None, sd_orig[2], sd_orig[3]]),
                       pd.Series([None] * 5, index=range(5,10)),
                                 pd.Series([sd_orig[4], sd_orig[5]], index=range(10,12)),
                                   pd.Series([None] * 8, index=range(13,21))])
# sd_orig = pd.concat([pd.Series([sd_orig[0], sd_orig[1], None, sd_orig[2], sd_orig[3]]),pd.Series([None] * 16, index=range(5,21))])
#pad (12,2) zeroes to original results
df_orig = pd.concat([df_orig, pd.DataFrame(0, index=range(12), columns=range(2))], axis=1)
#load zero results
df_zero = pd.read_csv(f'{results_path}/zero_results.csv', header=None)
mean_zero = df_zero.mean(axis=0)
sd_zero = df_zero.std(axis=0)
#load orthogonal results
df_orthog = pd.read_csv(f'{results_path}/orthog_results.csv', header=None)
mean_orthog = df_orthog.mean(axis=0)
sd_orthog = df_orthog.std(axis=0)
#load eval results
df_eval = pd.read_csv(f'{results_path}/eval_results.csv', header=None)
mean_eval = df_eval.mean(axis=0)
sd_eval = df_eval.std(axis=0)
#result variable names
rownames = ['train_v1','test_v1','recon_v1', 'train_v2', 'test_v2', 'recon_v2',
            'recon_train_v1', 'recon_test_v1', 'recon_train_v2', 'recon_test_v2', 
            'train_both', 'test_both', 'recon_both1', 'recon_both2', 'recon_both',
            'au1', 'au2', 'loss1','loss2', 'recon_loss1', 'recon_loss2']
#create results dataframe
results = pd.DataFrame({'mean_orig': mean_orig, 'sd_orig': sd_orig,
                        'mean_zero': mean_zero, 'sd_zero': sd_zero,
                        'mean_orthog': mean_orthog, 'sd_orthog': sd_orthog,
                        'mean_eval': mean_eval, 'sd_eval': sd_eval})
results.set_index([rownames], inplace=True)
results.to_csv(f'{results_path}/all_results.csv')

df_zero = pd.read_csv(f'{results_path}/zero_results.csv', header=None)
au_zero = (df_zero[15]+df_zero[16])*2.5
df_orthog = pd.read_csv(f'{results_path}/orthog_results.csv', header=None)
au_orthog = (df_orthog[15]+df_orthog[16])*2.5
df_eval = pd.read_csv(f'{results_path}/eval_results.csv', header=None)
au_eval = (df_eval[15]+df_eval[16])*2.5
results_au = pd.DataFrame({'mean_zero': au_zero.mean(), 'sd_zero': au_zero.std(),
                        'mean_orthog': au_orthog.mean(), 'sd_orthog': au_orthog.std(),
                        'mean_eval': au_eval.mean(), 'sd_eval': au_eval.std()}, index=range(1))
results_au.to_csv(f'{results_path}/au_results.csv')


df = pd.read_csv(f'{results_path}/all_results.csv')
df_sub = df.iloc[[1, 2,7, 4, 5, 9], 1:9]


df_sub.set_index([['test_v1','recon_v1', 'recon_test_v1','test_v2', 'recon_v2',
              'recon_test_v2']], inplace=True)

bar_width = 0.25
positions = jnp.arange(len(df_sub))

plt.rcParams['text.usetex'] = True
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
viridis = plt.cm.get_cmap('viridis', 3)  # 3 colors for 3 groups
colors = viridis(jnp.linspace(0, 1, 3))
# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
# plt.set_cmap('viridis')

# Plot each group with error bars
bar1 = ax.bar(positions - bar_width, df_sub['mean_zero'], yerr=df_sub['sd_zero'], capsize=5, width=bar_width, label=r'$\mathbf{C}=\mathbf{0}$',color=colors[0])
bar3 = ax.bar(positions , df_sub['mean_eval'], yerr=df_sub['sd_eval'], capsize=5, width=bar_width, label=r'$\sigma_1(\mathbf{C})<1$',color=colors[1])
bar2 = ax.bar(positions + bar_width, df_sub['mean_orthog'], yerr=df_sub['sd_orthog'], capsize=5, width=bar_width, label=r'$\mathbf{C}^T\mathbf{C}=\mathbf{C}\mathbf{C}^T=\mathbf{I}$', color=colors[2])
index_labels = [r'$\left(\tilde{X}_1,\tilde{X}_1\right)$', r'$\left(\tilde{X}_1,\tilde{X}_{1|2}\right)$', r'$\left(\tilde{X}_{1|2},\tilde{X}_{1|2}\right)$',
                r'$\left(\tilde{X}_2,\tilde{X}_2\right)$', r'$\left(\tilde{X}_2,\tilde{X}_{2|1}\right)$', r'$\left(\tilde{X}_{2|1},\tilde{X}_{2|1}\right)$']
# Customizations
ax.set_xticks(positions)
ax.set_xticklabels(index_labels)
ax.set_ylabel('Accuracy')
ax.set_xlabel('(Train dataset, Test dataset)')
# ax.set_title('Grouped Bar Chart with Standard Deviation Error Bars')
# ax.legend(title='Mean Type')
#place legend below plot
plt.legend(loc='lower right')
# Show the plot
plt.tight_layout()
plt.savefig('figures/accuracy_results.png')
plt.show()



df = pd.read_csv(f'{results_path}/all_results.csv')
df_sub = df.iloc[[11,13,12,14], 3:9]


df_sub.set_index([['test','test_v1imputed', 'test_v2imputed','both_imputed']], inplace=True)

bar_width = 0.25
positions = jnp.arange(len(df_sub))

plt.rcParams['text.usetex'] = True
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
viridis = plt.cm.get_cmap('viridis', 3)  # 3 colors for 3 groups
colors = viridis(jnp.linspace(0, 1, 3))
# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))
# plt.set_cmap('viridis')

# Plot each group with error bars
bar1 = ax.bar(positions - bar_width, df_sub['mean_zero'], yerr=df_sub['sd_zero'], capsize=5, width=bar_width, label=r'$\mathbf{C}=\mathbf{0}$',color=colors[0])
bar3 = ax.bar(positions , df_sub['mean_eval'], yerr=df_sub['sd_eval'], capsize=5, width=bar_width, label=r'$\sigma_1(\mathbf{C})<1$',color=colors[1])
bar2 = ax.bar(positions + bar_width, df_sub['mean_orthog'], yerr=df_sub['sd_orthog'], capsize=5, width=bar_width, label=r'$\mathbf{C}^T\mathbf{C}=\mathbf{C}\mathbf{C}^T=\mathbf{I}$', color=colors[2])
index_labels = [r'$[\tilde{X}_1;\tilde{X}_2]$', r'$[\tilde{X}_1;\tilde{X}_{2|1}]$',
                 r'$[\tilde{X}_{1|2};\tilde{X}_2]$',
                r'$[\tilde{X}_{1|2};\tilde{X}_{2|1}]$']
# Customizations
ax.set_xticks(positions)
ax.set_xticklabels(index_labels)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Test dataset')
# ax.set_title('Grouped Bar Chart with Standard Deviation Error Bars')
# ax.legend(title='Mean Type')
#place legend below plot
plt.legend(loc='lower right')
# Show the plot
plt.tight_layout()
plt.savefig('figures/accuracy_conc_results.png')
plt.show()
#done
