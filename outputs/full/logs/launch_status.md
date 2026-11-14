# Current status: completed

Full completed after approved memory optimization and explicit authorization to continue without the swap-growth threshold. See `../REPORT.md` and `../metrics/full_execution_manifest.json`. The following is the historical initial preflight decision, before optimization and training.

# Full launch: stopped before training

Hardware: 8 GiB RAM, 8 logical CPUs. Load averages: 1.98 / 2.34 / 2.56.
Swap: 1535.31 MiB used of 2048 MiB. Compressor occupies about 2.97 GiB.
Disk: about 13 GiB free, data volume 94% used.

The full preflight plans for 57,811 eligible users, 10,000 items and
370,990 training events (189,770 unique user-item pairs). These are train-only
selection counts, not completed experiment or evaluated-user counts.

The existing conservative ItemKNN intermediate-memory allowance is 7.45 GiB,
plus data frames, candidate features, retained models and the OS. It is not a
measured peak, but the current machine does not provide a safe reserve.

No model training or notebook execution was started. No full ranking metrics,
confidence intervals, model timings or measured experiment RAM peak exist.
No new figures were generated; future full exports must use dpi=300.
Dev artifacts, models, grids and notebook source remain unchanged.

Options requiring user agreement: (1) reduce peak memory through blockwise
sparse ItemKNN and lifecycle cleanup while keeping the full benchmark and
search spaces; validate equivalence on dev before reevaluating the resource
budget, (2) run full on a host with at least 32 GiB RAM, or (3) agree on a
smaller benchmark. No benchmark reduction or optimization was applied.
