# calibration_records — paper 1 release v1.3

Export only; nothing was rerun.

`<model>_grid.csv` — every node of the frozen calibration grid for that rival:
`node_index`, one column per parameter, `loss`, `selected`.  Node parameters
are `stage3_calibrate.py` `GRIDS`, enumerated in `itertools.product` order over
the axes in declaration order, which is the order the sweep sharded over.

**The `loss` column is empty at every node except the selected one.**  The
per-node scores were written to `s3cal/<model>_<shard>.npz` (fields `res`,
`names`, `scale`) and those shards were never promoted to artifacts: they are
absent from the artifact store (208 artifacts, no npz carrying a `res` field),
from `stage3_shards.tar.gz` (72 members, all `s3/` sweep shards for focal,
lag, two_state, rtip, ising), and from disk.  Only the selected node's loss
survives, in `null_db_results.npz` (`calib[2]`) for db and in the comparison
table of `null_db_memo.md` for the other four.  Recovering the full grids
requires rerunning `stage3_calibrate.py`, which this export does not do.

`selected_nodes.json` — per model: the selected parameter vector, its node
index and per-axis indices, the grid axes and shape, interior/boundary status,
the stored loss with its source, and the Table S1 comparison.

Objective: unweighted standardised SSE against the focal model's own output on
the 75 frozen training cells, `CAL_SEEDS = (3141, 3142)`.  No empirical data.
