* TOC
{:toc}

# List of all pages?

Will it work??

{% assign doclist = site.pages | sort: 'url'  %}
<ul>
    {% for doc in doclist %}
        {% if doc.name contains '.md' or doc.name contains '.html' %}
            <li><a href="{{ site.baseurl }}{{ doc.url }}">{{ doc.url }}</a></li>
        {% endif %}
    {% endfor %}
</ul>

# hello world

hello world

## Section 2

here is a little code snippet:

```rust
use dfdx::prelude::*;

trait MyCoolTrait {
    type Something;
    fn do_it(&self) -> Self::Something;
}

fn do_something<const K: usize, T>(a: f32, t: T) -> T {
    let _ = T::NUM_ELEMENTS;
    todo!();
}

fn a_lifetime_function<'a>(something: &'a f32) {
    println!("{:?}", a);
}

fn main() {
    // this is a comment
    let q: Tensor2D<3, 5, OwnedTape> = TensorCreator::zeros();
    let r = q.sin() + 1.2345;
    println!("hello world");
    a_lifetime_function::<'static>(&-0.234);
}
```

### Section 3

> here's another code snippet

### Another section

[Link to another page](2022-09-10_test-page.md)
