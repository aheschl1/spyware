"""Pipeline-specific queries — one module per processing pipeline.

Pipelines are responsible for their own querying: how a pipeline finds work
and reads its inputs is its own business, and those queries live here rather
than in the generic repositories. Convention:

- one module per pipeline, named after it;
- classes extend :class:`~database.repos.base.BaseRepo`;
- the pipeline constructs its repo itself (``SessionStatsQueries(pipe.connection)``)
  — these are never hung off :class:`~database.pipe.DatabasePipe`, which stays
  generic.
"""
