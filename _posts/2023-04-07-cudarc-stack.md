---
title: "A look into how dfdx compiles & uses cuda kernels"
---

# Table of contents

Join the [dfdx discord](https://discord.gg/HwaNMuHa) to stay in the loop!

GPU Acceleration with CUDA was recently added into dfdx. In this post I'll dive into:

1. Each step of the CUDA pipeline, from compiling cuda kernels to how launching them works
2. How generics are handled across FFI
3. How all of this is tested in a scalable way

* TOC
{:toc}

# What is a CUDA kernel?

Basically: a function. Behind the scenes it is executed on massively GPUs that are organized into thousands of groups of threads. Here's what a simple cuda kernel looks like in code:

```c++
extern "C" __global__ void sin_kernel(const size_t n, const float *inp, float *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sinf(inp[i]);
    }
}
```

That's all the introduction I'll give for now, as the rest of this post just focuses on how to interact with these from rust. The details of writing kernels is for another time.

# How do you use them from not c++?

Most of the CUDA examples you'll see will invoke these kernels directly in c++, often with these crazy bracket symbols:
```c++
sin_kernel<<<12, 34>>>(12, ...)
```

But since we are using rust, we can't do that. Instead we'll go through nvidia's CUDA toolkit api to load a module with [cuModuleLoadData](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b), which states:

> Takes a pointer image and loads the corresponding module module into the current context. The pointer may be obtained by mapping a cubin or PTX or fatbin file, **passing a cubin or PTX or fatbin file as a NULL-terminated text string**, or incorporating a cubin or fatbin object into the executable resources and using operating system calls such as Windows FindResource() to obtain the pointer. 

This means if we can compile these cuda kernels into a PTX file, then we can give the content of the PTX to this function! Now we just need to figure out how to compile these PTX objects.

# Compiling CUDA Kernels

There are two methods for compiling `.cu` files to `.ptx` files:
1. Use `nvcc`, which is nvidia's cuda compiler
2. Use `nvrtc`, which is nvidia's runtime compiler

`dfdx` actually uses both methods in the codebase in different situations. Let's dive into them!

## At compile time with `nvcc`

There are two problems to solve here:

1. How do we invoke nvidia's kernel compiler `nvcc` at compile time?
2. How do we gain access to the output in the source code?

### Running `nvcc` in a build script

The answer to both of these is a build script (`build.rs`)! It turns out to be quite simple actually; dfdx's `build.rs` file does the following:
1. Glob for all `*.cu` files in the dfdx repo.

    ```rust
    let kernel_paths: Vec<std::path::PathBuf> = glob::glob("src/**/*.cu")
        .unwrap()
        .map(|p| p.unwrap())
        .collect();
    ```

2. Invokes nvidia's kernel compiler `nvcc` on each individual cu file using `std::process`.
    1. These are saved into the `$OUT_DIR` directory, which is a [special build.rs environment variable](https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts).
    2. The saved file's extension will be `.ptx` (important for later).

    ```rust
    kernel_paths
        .iter()
        .map(|kernel_path| {
            std::process::Command::new("nvcc")
                .arg("--ptx")
                .args(["--output-directory", &std::env::var("OUT_DIR").unwrap()])
                .arg(kernel_path)
                .spawn()
                .unwrap()
        })
        .collect::<Vec<_>>()
    ```

That's it for compiling ptx files!

### Inserting file contents into rust source code

As far as using the output of these, you'll find this snippet all around the dfdx cuda kernels:

```rust
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/conv2d.ptx"));
```

Each piece of this is important, so starting from the inner most out:
1. [env!(...)](https://doc.rust-lang.org/std/macro.env.html) "Inspects an environment variable at compile time.", meaning it will have the same value of `$OUT_DIR` that was used in `build.rs`
2. [concat!(...)](https://doc.rust-lang.org/std/macro.concat.html) concatenates two string literals. So the result of this is `"$OUT_DIR/conv2d.ptx"` in this case.
    a. Since we know the saved file is just the name of the `.cu` file replaced with `.ptx`, we know this will exist.
3. [include_str!(...)](https://doc.rust-lang.org/std/macro.include_str.html) includes the contents of a file **at compile time**. This means we are embedding the compiled `.ptx` file into our rust source code!

To actually load this ptx into the device, you'll see code like:
```rust
self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
```

Which converts the `PTX_SRC` into a [cudarc::nvrtc::Ptx](https://docs.rs/cudarc/0.9.6/cudarc/nvrtc/safe/struct.Ptx.html) object, and then uses [cudarc::driver::CudaDevice::load_ptx](https://docs.rs/cudarc/0.9.6/cudarc/driver/safe/struct.CudaDevice.html#method.load_ptx) to load this module into Cuda.

## JIT compiling at run-time with `nvrtc`

We can also JIT compile kernels at runtime. This path is actually much easier code wise, but the first invocation of the kernel will involve compiling as well.

To show off why you might go with this path instead of nvcc, I'll use a special kernel as an example. This kernel is the equivalent of the `as` keyword in rust: it converts between types! Notably, it has to support **every** pair of types that we might want to run with. This is super cumbersome if we go with nvcc due to generics as I'll get into later.

`nvrtc` is powerful, because we can have "pseudo" generic kernels that we compile for each type we need at runtime.

Here's the source code for the kernel as a rust global const. Note the types of the input:

```rust
const AS_KERNEL: &str = "
extern \"C\" __global__ void as_kernel(const size_t n, const $Src *inp, $Dst *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = inp[i];
    }
}";
```

At runtime, we can dynamically create and load the module into Cuda like so, where `E1` is the source's type, and `E2` is the destination type.
```rust
let module_name = std::format!("convert_{}_to_{}", E1::NAME, E2::NAME);
if !dev.has_func(&module_name, "as_kernel") {
    let src = AS_KERNEL.replace("$Src", E1::NAME).replace("$Dst", E2::NAME);
    let ptx = compile_ptx(src).unwrap();
    dev.load_ptx(ptx, &module_name, &["as_kernel"])?;
}
```

A number of things are going on here:
1. We *only* compile it if the device doesn't already have the module loaded
2. The types are determined at runtime & pasted into the kernel's source
3. We use [cudarc::nvrtc::compile_ptx](https://docs.rs/cudarc/0.9.6/cudarc/nvrtc/safe/fn.compile_ptx.html) to turn this into a `Ptx` object.
4. We load it into cuda with `.load_ptx`

# Launching cuda kernels

Now that we know how to compile & load a PTX file into a cuda device, we can actually submit this kernel to the GPU!

1. Retrieve the already loaded function:
```rust
let as_kernel = dev.get_func(&module_name, "as_kernel").unwrap();
```

2. Allocate space for output - we are specifically not setting the memory here for performance reasons and because the kernel we are about to launch will set all the memory.
```rust
let n: usize = inp.data.len();
let mut out = unsafe { dev.alloc::<E2>(n) }?;
```

3. Configure the grid/block dimensions for cuda & the arguments for the kernel
```rust
// set up the arguments to the function to pass to the kernel
let cfg = launch_cfg(n as u32);
let args = (
    n,                 // `usize` <=> `const size_t n`
    inp.data.as_ref(), // `&CudaSlice<E1>` <=> `const E1 *inp`
    &mut out           // `&mut CudaSlice<E2>` <=> `E2 * out`
);
```

4. Launch it!
```rust
// Asynchronously launch the kernel
unsafe { as_kernel.launch(cfg, args) }?;
```

# Generics across FFI

`dfdx` heavily uses generics everywhere. The data type of tensors (dtype) is a generic type, meaning all the low level kernels are also generic over the dtype. We also know that c++ has generics via templates. Should be easy to use on both sides (rust and c++) right? **Nope!** Since we are using FFI, we need to annotate all kernel methods with `extern "C"`. You actually [cannot use `extern "C"` together with templates](https://stackoverflow.com/questions/4877705/why-cant-templates-be-within-extern-c-blocks)!

To be clear, you can write the following kernel, but the name of this function will be mangled because of the typename:
```c++
template<typename T>
__global__ void kernel(const T *inp, T *out, int numel) { ... }
```

To use with FFI in rust, you need to add `extern "C"` to all the kernels.

There are two ways to handle this:

1. Add an `extern "C"` kernel that calls into a generic kernel for each data type you need
2. JIT compile a kernel with the types inserted in at runtime.

The first option looks like the below on the cuda side:
```c++
template<typename T>
__device__ void kernel(const T *inp, T *out, int numel) { ... }

extern "C" __global__ void kernel_f32(const float *inp, float *out, int numel) { kernel(inp, out, numel) }
extern "C" __global__ void kernel_f64(const double *inp, double *out, int numel) { kernel(inp, out, numel) }
```

This works for a majority of cases, but is very repetitive.

I'll discuss the second option more below.


# Testing

So how do we test all this? `dfdx` already had extensive unit testing on all tensor operations, but this essentially doubled the amount of code to cover for each operation (because we have Cpu kernels and Cuda kernels). Luckily we still have generics!

`dfdx` has a special `TestDevice` type that can be configured with feature flags:

```rust
#[cfg(not(feature = "cuda"))]
pub type TestDevice = crate::tensor::Cpu;

#[cfg(feature = "cuda")]
pub type TestDevice = crate::tensor::Cuda;
```

As long as the unit tests are written using this TestDevice type, we can easily switch between what type of kernel we are testing. This is super useful because it forces all the unit tests to be written in a generic way, which is much closer to how I expect users to write code as well.

Here's an example of the `square` operation which square's every value:

```rust
#[test]
fn test_square() {
    let dev: TestDevice = Default::default();
    let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
    let r = x.leaky_trace().square();
    assert_eq!(r.array(), [4.0, 1.0, 0.0, 1.0, 4.0]);
    let g = r.mean().backward();
    assert_eq!(g.get(&x).array(), [-0.8, -0.4, 0.0, 0.4, 0.8]);
}
```

You also might notice a `TestDtype` type which can be used to test `f64` instead of `f32`.

# Thanks

A big thanks to all my sponsors, including: @jackvial, @TimerErTim, @paholg, @scooter, @zeroows, @iExalt, @quietlychris, @skinner, @Michael P, @Alex R., @rahul-tuli.
