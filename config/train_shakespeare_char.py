# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 20 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

# use of flash attention
use_flash_attention = True

wandb_log = True # override via command line if you like
wandb_project = 'profile-nano-gpt-shakespeare' 

if use_flash_attention:
    wandb_run_name = 'flash-attention'
else:
    wandb_run_name = 'slow-attention'

dataset = 'shakespeare_char' 
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000 #MM:5000
lr_decay_iters = max_iters # 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 50 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model


#nsys profile --show-output=true --gpu-metrics-device=0 --gpu-metrics-frequency=10 --gpu-metrics-set=0 --trace=cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=kernel --cuda-memory-usage=true --stop-on-exit=true -o flash_profile python train.py config/train_shakespeare_char.py


         