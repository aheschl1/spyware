"""Concrete pipelines.

Import discipline: pipeline modules import only the stdlib, ``processing.base``,
``database.*``, and ``storage.*`` at module top. Anything heavy (ML models,
inference clients) is imported and constructed inside ``setup()`` or
``process()`` — the registry, and everything that imports it, must stay cheap
to load.
"""
