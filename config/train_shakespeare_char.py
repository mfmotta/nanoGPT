# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

#I/O
out_dir = 'out-shakespeare-char'
log_interval = 20 # don't print too too often

# data
dataset = 'shakespeare_char' 
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2 
bias = False # TODO do we use bias inside LayerNorm and Linear layers?

# use of flash attention
flash = False

# adamw optimizer 
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
weight_decay = 1e-1
max_iters = 400
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

# NSight profiling: at which iteration start and end profiling
profiling_start = 2 * warmup_iters
profiling_end   = profiling_start + 3

#model initialization
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# logging
wandb_log = True # override via command line if you like
wandb_project = 'profile-attention-nano-gpt' 
if flash: #use flash attention
    wandb_run_name = 'flash-attention'
else:
    wandb_run_name = 'slow-attention'


         