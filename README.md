Welcome! This is my blog on my rust crate [dfdx](https://github.com/coreylowman/dfdx).

You can find the documentation at [docs.rs/dfdx](https://docs.rs/dfdx/).

# Latest Posts

- [002.md](002.md)
- [001.md](001.md)
- [2022-09-10_test-page.md](2022-09-10_test-page.md)

# All Posts

{% assign doclist = site.pages | sort: 'url'  %}
<ul>
    {% for doc in doclist %}
        {% if doc.name contains '.md' %}
            <li><a href="{{ site.baseurl }}{{ doc.url }}">{{ doc.url }}</a></li>
        {% endif %}
    {% endfor %}
</ul>
