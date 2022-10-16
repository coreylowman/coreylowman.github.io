Welcome! This is my blog on my rust crate [dfdx](https://github.com/coreylowman/dfdx).

You can find the documentation at [docs.rs/dfdx](https://docs.rs/dfdx/).

# Latest Posts

- [002.md](002.md)
- [001.md](001.md)
- [2022-09-10_test-page.md](2022-09-10_test-page.md)

# All Posts

<ul>
    {% for page in site.posts %}
        {% if page.name contains '.md' %}
            <li><a href="{{page.url}}">{{page.title}}</a></li>
        {% endif %}
    {% endfor %}
</ul>
