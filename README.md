Welcome! This is my blog on my rust crate [dfdx](https://github.com/coreylowman/dfdx).

# What is dfdx?

dfdx is a deep learning library built entirely in rust. With it you can create deep neural networks
and optimize them. dfdx's apis know the shapes of tensors *at compile time*, so every
operation you do will be checked by the compiler. *No more mismatched shape errors!*

You can find more details at:
1. [github](https://github.com/coreylowman/dfdx)
2. [docs.rs/dfdx](https://docs.rs/dfdx/latest/dfdx/)
3. [crates.io/dfdx](https://crates.io/crates/dfdx)

# Posts

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
