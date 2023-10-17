# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
import random

# data
dataset = 'shakespeare_char' 
gradient_accumulation_steps = 1 
batch_size = 64
block_size = 256 #=sequence length= context of up to 256 previous characters

# model
n_layer = 6
n_head = 6
n_embd = 384 #head dimension
dropout = 0.2 
bias = False # TODO do we use bias inside LayerNorm and Linear layers?

# use of flash attention
flash = True

# adamw optimizer 
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
weight_decay = 1e-1
max_iters = 100
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

eval_interval = max_iters//4 # keep frequent because we'll overfit
eval_iters = max_iters//5
eval_only = False # if True, script exits right after the first eval

# learning rate decay settings
warmup_iters = max(10, max_iters // 100) # not super necessary potentially
min_lr = 1e-4 # learning_rate / 10 usually
decay_lr = True # whether to decay the learning rate
lr_decay_iters = max_iters # make equal to max_iters usually


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

# NSight profiling: at which iteration start and end profiling
profiling_start = 2 * warmup_iters
profiling_end   = profiling_start + 3

#model initialization
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# logging
wandb_log = True # override via command line if you like
wandb_project = 'profile-attention-nano-gpt' 
#parameters ='-n_iters'+str(max_iters)+'-n_head'+str(n_head)+'-h_size'+str(n_embd)+'-seq_len'+str(block_size)
#if flash: #use flash attention
   # wandb_run_name = 'flash'+parameters
#else:
   # wandb_run_name = 'slow'+parameters


# I/O
out_dir = 'out-shakespeare-char'
log_interval = max_iters // 10 # don't print too too often
random_port = 1024 + random.randint(1,1000)

