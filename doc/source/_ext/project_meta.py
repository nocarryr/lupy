from __future__ import annotations
from typing import NamedTuple, Sequence, Any, TYPE_CHECKING

from pathlib import Path
import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

from docutils import nodes, utils
from sphinx.util.nodes import split_explicit_title
from sphinx import addnodes
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from docutils.nodes import Node, system_message
    from docutils.parsers.rst.states import Inliner

    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata


PROJ_META: ProjMeta|None = None

class ProjMeta(NamedTuple):
    name: str
    version: str
    authors: list[dict[str, str]]
    description: str
    readme: str
    keywords: list[str]
    dependencies: list[str]
    urls: dict[str, str]

    @property
    def parsed_version(self) -> Version:
        return Version(self.version)


def find_pyproj(cur_dir: Path|None = None) -> Path:
    if cur_dir is None:
        cur_dir = Path(__file__).resolve().parent
    fn = cur_dir / 'pyproject.toml'
    if fn.exists():
        return fn
    try:
        parent = cur_dir.parents[0]
    except IndexError:
        raise FileNotFoundError('Could not find "pyproject.toml"')
    return find_pyproj(parent)


def get_proj_meta() -> ProjMeta:
    global PROJ_META
    meta = PROJ_META
    if meta is None:
        filename = find_pyproj()
        meta = PROJ_META = load_proj_meta(filename)
    return meta

def load_proj_meta(filename: Path) -> ProjMeta:
    data = tomllib.loads(filename.read_text())
    kw = {k: data['project'][k] for k in ProjMeta._fields}
    return ProjMeta(**kw)




def project_url_role(typ: str, rawtext: str, text: str, lineno: int,
    inliner: Inliner, options: dict[str, Any] | None = None,
    content: Sequence[str] = (),
) -> tuple[list[Node], list[system_message]]:
    text = utils.unescape(text)
    has_explicit_title, title, part = split_explicit_title(text)
    meta = get_proj_meta()
    url = meta.urls[part]
    if not has_explicit_title:
        title = url
    pnode = nodes.reference(title, title, internal=False, refuri=url)
    return [pnode], []


class ProjectUrlDirective(SphinxDirective):
    required_arguments = 1

    def run(self) -> list[Node]:
        key = self.arguments[0]
        meta = get_proj_meta()
        url = meta.urls[key]
        inline = nodes.inline('', url)
        onlynode = addnodes.only(expr='html')
        onlynode += nodes.reference('', '', inline, internal=False, refuri=url)
        return [onlynode]


class ProjectDependencies(SphinxDirective):
    required_arguments = 0

    def run(self) -> list[Node]:
        meta = get_proj_meta()
        list_node = nodes.bullet_list()
        for dep in meta.dependencies:
            req = Requirement(dep)
            if req.marker is not None:
                txt = f'{req.name} (for {req.marker})'
            else:
                txt = req.name

            item = nodes.list_item()
            list_node += item
            item += nodes.paragraph(text=txt)
        return [list_node]



def update_config(app: Sphinx) -> None:
    meta = get_proj_meta()
    v = meta.parsed_version
    conf = {
        'project': meta.name,
        'version': f'{v.major}.{v.minor}',
        'release': str(v)
    }
    for key, val in conf.items():
        conf_val = app.config[key]
        if conf_val == app.config.config_values[key].default:
            app.config[key] = val


def setup(app: Sphinx) -> ExtensionMetadata:
    app.add_directive('project-dependencies', ProjectDependencies)
    app.add_role('project-url', project_url_role)

    update_config(app)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
