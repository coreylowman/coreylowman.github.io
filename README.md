Welcome! This is my blog on my rust crate [dfdx](https://github.com/coreylowman/dfdx).

You can find the documentation at [docs.rs/dfdx](https://docs.rs/dfdx/).

# Posts

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
