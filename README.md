This repo is based on the official Stable Diffusion [repo](https://github.com/CompVis/stable-diffusion "repo") and its [variants](https://github.com/basujindal/stable-diffusion "variants"), enabling running stable-diffusion on GPU with only 1GB VRAM.

To reduce the VRAM usage, the following opimizations are used:
- Based on [PTQD](https://github.com/ThisisBillhe/PTQD), the weights of diffusion model are quantized to 2-bit, which reduced the model size to only 369M (only diffusion model are quantized, not including the cond_stage_model and first_stage_model).
- The stable diffusion model is fragmented into four parts which are sent to the GPU only when needed. After the calculation is done, they are moved back to the CPU.
- The attention calculation is done in parts.


<h1 align="center">Installation</h1>

Establish a virtual environment and install dependencies as referred to the official [repo](https://github.com/CompVis/stable-diffusion "repo").
The quantized model checkpoint can be downloaded from [Google drive](https://drive.google.com/file/d/1bdsW5Bys70xt3x4DDxNKsbMkRkkKgneJ/view?usp=drive_link)

<h1 align="center">Usage</h1>

Only txt2img is supported now.
## txt2img

- `txt2img` can generate _512x512 images from a prompt using under 1GB GPU VRAM (evaluated with pytorch2.0 on RTX3090).

- For example, the following command will generate 10 512x512 images:

`python3 tiny_optimizedSD/tiny_txt2img.py --prompt "A peaceful lakeside cabin with a dock, surrounded by tall pine trees and a clear blue sky" --H 512 --W 512 --seed 27`

<h1 align="center">Arguments</h1>

## `--seed`

**Seed for image generation**, can be used to reproduce previously generated images. Defaults to a random seed if unspecified.

- The code will give the seed number along with each generated image. To generate the same image again, just specify the seed using `--seed` argument. Images are saved with its seed number as its name by default.

- For example if the seed number for an image is `1234` and it's the 55th image in the folder, the image name will be named `seed_1234_00055.png`.

## `--n_samples`

**Batch size/amount of images to generate at once.**

- To get the lowest inference time per image, use the maximum batch size `--n_samples` that can fit on the GPU. Inference time per image will reduce on increasing the batch size, but the required VRAM will increase.

- If you get a CUDA out of memory error, try reducing the batch size `--n_samples`. If it doesn't work, the other option is to reduce the image width `--W` or height `--H` or both.

## `--n_iter`

**Run _x_ amount of times**

- Equivalent to running the script n_iter number of times. Only difference is that the model is loaded only once per n_iter iterations. Unlike `n_samples`, reducing it doesn't have an effect on VRAM required or inference time.

## `--H` & `--W`

**Height & width of the generated image.**

- Both height and width should be a multiple of 64.

## `--turbo`

**Increases inference speed at the cost of extra VRAM usage.**

- Using this argument increases the inference speed by using around 700MB of extra GPU VRAM. It is especially effective when generating a small batch of images (~ 1 to 4) images. It takes under 20 seconds for txt2img and 15 seconds for img2img (on an RTX 2060, excluding the time to load the model). Use it on larger batch sizes if GPU VRAM available.

## `--precision autocast` or `--precision full`

**Whether to use `full` or `mixed` precision**

- Mixed Precision is enabled by default. If you don't have a GPU with tensor cores (any GTX 10 series card), you may not be able use mixed precision. Use the `--precision full` argument to disable it.

## `--format png` or `--format jpg`

**Output image format**

- The default output format is `png`. While `png` is lossless, it takes up a lot of space (unless large portions of the image happen to be a single colour). Use lossy `jpg` to get smaller image file sizes.

## `--unet_bs`

**Batch size for the unet model**

- Takes up a lot of extra RAM for **very little improvement** in inference time. `unet_bs` > 1 is not recommended!

- Should generally be a multiple of 2x(n_samples)

<h1 align="center">Weighted Prompts</h1>

- Prompts can also be weighted to put relative emphasis on certain words.
  eg. `--prompt tabby cat:0.25 white duck:0.75 hybrid`.

- The number followed by the colon represents the weight given to the words before the colon. The weights can be both fractions or integers.

## Troubleshooting

### Green colored output images

- If you have a Nvidia GTX series GPU, the output images maybe entirely green in color. This is because GTX series do not support half precision calculation, which is the default mode of calculation in this repository. To overcome the issue, use the `--precision full` argument. The downside is that it will lead to higher GPU VRAM usage.

