"""The explicit resource registry.

Adding a resource is: write the class, import it here, append an instance to
``RESOURCES``. Mirrors ``processing/registry.py``.
"""

from resources.audio import AudioResource
from resources.base import ResourceType
from resources.location import LocationResource

RESOURCES: tuple[ResourceType, ...] = (
    AudioResource(),
    LocationResource(),
)

_BY_NAME = {resource.name: resource for resource in RESOURCES}


def names() -> tuple[str, ...]:
    return tuple(resource.name for resource in RESOURCES)


def get(name: str) -> ResourceType:
    """The registered resource type (KeyError if unknown)."""
    return _BY_NAME[name]


if len(_BY_NAME) != len(RESOURCES):
    raise RuntimeError("duplicate resource names in RESOURCES")
if "audio" not in _BY_NAME:
    # The wire default: an old-protocol chunk header carries no resource field.
    raise RuntimeError("the 'audio' resource must exist")
for _resource in RESOURCES:
    if not _resource.name or not _resource.name.replace("-", "").isalpha() or _resource.name != _resource.name.lower():
        raise RuntimeError(f"resource name {_resource.name!r} must be lower-kebab")
