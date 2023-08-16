# Experiments with [nanoGPT](https://github.com/karpathy/nanoGPT)

Here we compare training  the nanoGPT model with 'slow' and 'flash' attention.

</br>

Results can be visualized [here](https://api.wandb.ai/links/m-motta/ky8ak8xl).


Notice how the training and validation metrics don't differ for both types of attention, but the GPU utilization differs significantly.

# Training parameters:

- gradient_accumulation_steps = 1
- batch_size = 64
- block_size = 256 (context of up to 256 previous characters)
- n_layer = 6
- n_head = 6
- n_embd = 384
- dropout = 0.2
- flash = True
- learning_rate = 1e-3
- max_iters = 6000
- lr_decay_iters = max_iters
- min_lr = 1e-4
- beta2 = 0.99

# GPU Profiling

We profile the forward and backward steps with NSight Systems with the following choices:

```shell
nsys profile --show-output=true --gpu-metrics-device=0 --gpu-metrics-frequency=10 --gpu-metrics-set=0 /
--trace=cuda,nvtx,cublas --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=kernel /
--cuda-memory-usage=true --stop-on-exit=true -o slow_profile python train.py config/train_shakespeare_char.py
```

