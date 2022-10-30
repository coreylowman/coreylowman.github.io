---
title: "Announcing dfdx - an deep learning library built with const generics"
---

Links: [github](https://github.com/coreylowman/dfdx), [crates.io/dfdx](https://crates.io/crates/dfdx), [docs.rs/dfdx](https://docs.rs/dfdx/latest/dfdx/)

Discord: [https://discord.gg/AtUhGqBDP5](https://discord.gg/AtUhGqBDP5)

Hey Everyone,

I've been working on dfdx for a while, and I want to get it out there now! As you'll see from the length of this post, there are so many details I want to share about how this library works.

It started out as a personal side project quite a while ago, but the design I ended up with made it really easy to get a ton of features.

# What is dfdx?

Short version: pytorch/tensorflow, but 100% in rust and really easy to use.

Long version: A deep learning library that contains tensors (up to 4d), tensor operations, backprop/auto diff implementation, neural network building blocks, and deep learning optimizers. It's fully implemented in rust, and uses const generics as much as possible (e.g. tensors have shapes known at compile time).

See github/docs.rs for more details.

# Why should I use it?

The DL ecosystem in rust is still pretty nascent. The crates I've seen were either wrappers around c++ libraries, not using const generics, or seemed kinda hard to use.

While building dfdx and looking at other DL libraries outside of rust, I also realized that many of them are very complicated and hard to understand. Pytorch for example has so many layers of indirection involved with the c++ side, its hard to find where things actually happen.

*A "fun" exercise: find the c++ binary cross entropy implementation in* [*pytorch*](https://github.com/pytorch/pytorch)*.*

I'm building dfdx to be:

1. Really easy to use/read (check out [examples/](https://github.com/coreylowman/dfdx/tree/main/examples))
2. Really easy to maintain/extend/understand (check out some internals in [src/tensor\_ops](https://github.com/coreylowman/dfdx/tree/main/src/tensor_ops))
3. Checked at compile time as much as possible (e.g. you can't call backward without gradients; operations require compatible tensor sizes)
4. Highly tested & verified against existing libraries

You can also save neural networks as `.npz` files which are pretty easy to load into pytorch, if you want to train in rust and then use in python 😃.

More details in github/docs.rs.

# How does it work?

The special bits are all in how the backward operations & derivatives are recorded on the `GradientTape`. For any given operation on a tensor or multiple tensors, there is (usually) only 1 resulting tensor. The `GradientTape` from the inputs is always moved to the resulting tensor. So a tensor operation in dfdx does the following:

1. Compute the result
2. Remove the tape from the input
3. Insert the backward operation into the tape
4. Put the tape into the result

Now there are actually two kinds of tapes in dfdx:

1. `OwnedTape`, which just a wrapper around `GradientTape`.
2. `NoneTape`, which is a unit struct and does not track backward operations.

To actually kick off gradient tracking, you need to actually explicitly insert a new `OwnedTape` into the input using `.trace()`! Then the `OwnedTape` will be carried through the entire forward pass tracking gradients. At the end when you call `backward()`, it removes the tape from the loss tensor, and runs all the operations in reverse to produce a `Gradients`. Then you go through the normal update of parameters with an `Optimizer`.

Since the type of `Tape` is tracked as a generic parameter of all tensors, all the operations know at compile time whether they are getting a `OwnedTape` or a `NoneTape`!

There's soooo much more to get into, and a lot of fun things about the implementation. See [README.md#fun-implementation-details](https://github.com/coreylowman/dfdx#funnotable-implementation-details) for some more tidbits.

# But is it faster than pytorch?

With all of my cpu laptop testing: yes. I've been doing all my speed benchmarking with [examples/mnist\_classifier.rs](https://github.com/coreylowman/dfdx/blob/main/examples/mnist_classifier.rs), and dfdx can be anywhere from x2-3 faster than pytorch is. I suspect a lot of this comes from optimizations rust can do since it has:

1. access to the entire program (pytorch libraries are compiled and then used by python and have to support all usescases)
2. const generic tensors (again pytorch can't do this since the c++ is a backend for python)
   1. fun fact: rust auto vectorizes a lot of the array operations, so dfdx contains no simd code!

I'll be adding more documentation and actual benchmarks in the future. [issue #20](https://github.com/coreylowman/dfdx/issues/20)

A nice/funny aside that shows dfdx's potential: pytorch recently posted [A BetterTransformer for Fast Transformer Inference](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/?utm_source=social&utm_medium=pytorch&utm_campaign=organic&utm_content=linkedin) on their blog, about speeding up transformers with "fastpath" execution (where gradients aren't tracked). In dfdx this would be trivial since you can just implement forward different for \`OwnedTape\` and \`NoneTape\`!

# What's missing?

Unfortunately, some important functionality is gated behind [feature(generic\_const\_exprs)](https://github.com/rust-lang/rust/issues/76560). See [dfdx issues](https://github.com/coreylowman/dfdx/issues?q=is%3Aopen+is%3Aissue+label%3Afeature%28generic_const_exprs%29). This includes:

* Convolutional layers. (The output size can be calculated at compile time, which requires doing simple arithmetic with const usize's)
* tensor flatten/reshapes/splitting/concatenation. (also requires arithmetic with const usize's).

I've been working a bit on nightly to test how all of this would work, but its quite unwieldy implementation wise at the moment.

I also have not added GPU support [issue #9](https://github.com/coreylowman/dfdx/issues/9). I think Rust-CUDA could be used for this, but this will probably require a ton of effort (I'm available for sponsorship if you really want this feature!).

Regardless of no GPU support, dfdx can [link to Intel MKL](https://github.com/coreylowman/dfdx#blas-libraries) which is really fast on the CPU!

# What's next?

I'm still discovering optimizations for speed/allocations in internal code, so I'm sure there'll be more of that. There's also plenty more optimizers/neural network layers/operations that are missing.

The biggest thing I'm working on next is Transformers [issue #34](https://github.com/coreylowman/dfdx/issues/34), which I do think dfdx can support without const generics.

As you can guess I could go on and on about dfdx, so I'm happy to answer any questions you have!