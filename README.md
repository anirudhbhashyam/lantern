# Goose tensors and MNIST diffusion

We implement a CPU tensor library in Goose.

## Run it

This was made on macos and was tested using the bundled Goose v2026.4 compiler on macOS arm64.

```sh
./run.sh test
./run.sh all
./run.sh sample 47
```

The steps can also run separately: `./run.sh data` downloads and checks the
two MNIST training archives, and `./run.sh train 24000` builds and trains from
scratch.

## Tensor API

```goose
import tensor from "./lib/tensor";

fn main() {
    let a = tensor::from_data<f32>([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v = tensor::view(a);
    let gram = tensor::matmul(v, tensor::transpose(v, 0, 1));
    print(gram.data); // [14.0, 32.0, 32.0, 77.0]
}
```

| Area | API |
| --- | --- |
| Construction | `full`, `zeros`, `ones`, `from_data`, `randn` |
| Layout | `view`, `at`, `set`, `slice`, `transpose`, `permute`, `reshape`, `broadcast_to`, `copy` |
| Math | `map`, `zip`, `add`, `sub`, `mul`, `div`, `relu`, `exp_t` |
| Reductions | `reduce`, `sum_dim`, `max_dim`, `mean_dim`, `sum_all`, `mean_all`, `softmax` |
| Linear algebra | `matmul`, `batched_matmul`, `matmul_generic`, `gemm_into` |
| Training | `linear_init`, `linear_forward`, `linear_backward`, `relu_backward_into`, `mse_backward`, `adam_step` |

We tried to follow what torch does.

## Model and sampling

The denoiser takes 784 noisy pixels, a one-hot digit class, and a sinusoidal
time embedding. Three 512-unit ReLU layers feed a 784-value prediction of the
clean image. Training minimizes mean squared error with Adam over batches of
128 from the 60,000 MNIST training images.
