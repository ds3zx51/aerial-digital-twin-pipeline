# Contributing

Thank you for helping improve the project. Before opening a pull request:

1. keep changes scoped and preserve the non-overwrite result policy;
2. do not add credentials, private infrastructure, raw results, models, or
   scene geometry;
3. do not tune AODT-only values, normalization, hyperparameters, or thresholds
   to make a backend comparison look closer;
4. update lineage and semantic qualifications when a data field changes; and
5. run `python -m unittest discover -s tests -v`.

Use feature branches and describe the scientific claim affected by the
change. Online AODT and GPU runs are never required for an ordinary code
review; attach only scrubbed summaries and hashes when integration evidence is
needed.

Contributions cannot be accepted under an open-source grant until the project
owner resolves [LICENSE_SELECTION_REQUIRED.md](../LICENSE_SELECTION_REQUIRED.md).
