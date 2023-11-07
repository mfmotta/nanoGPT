import time
import random

# data
dataset = '/home/nanoGPT/finetune/data' #TODO fix path to dataset
gradient_accumulation_steps = 8 #32
batch_size = 1
#block_size = 256 #=sequence length= context of up to 256 previous characters

#model
init_from = 'gpt2-xl' # this is the largest GPT-2 model
#this defines: ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size'] #, 'flash']:
flash =  True
# params below should be overriden #TODO check
n_layer = 6
n_head = 6
n_embd = 384 #head dimension
dropout = 0.2 
bias = False # TODO do we use bias inside LayerNorm and Linear layers?
####
out_dir = 'out-finetune'
eval_interval = 5
eval_iters = 40
wandb_log = False 
wandb_project = 'finetune'
wandb_run_name = 'ft-' + str(time.time())


# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
max_iters = 40

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
weight_decay = 1e-1

beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

warmup_iters = max(10, max_iters // 100)

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

# PyTorch profiler schedule arguments:
profiler_schedule_args = {
    'skip_first': 5,
    'wait': warmup_iters // 2,
    'warmup': warmup_iters // 2,
    'active': 3,
    'repeat': 1
}

random_port = 1024 + random.randint(1,1000) 
