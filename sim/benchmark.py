"""Repeat the mission with different sensor-noise seeds and summarise.

    python -m sim.benchmark --runs 10
"""

import argparse

from sim.run_simulation import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    rows = []
    print("seed | mission time (s) | map coverage | red tiles (z1, z2) | collisions | border crossed | returned home")
    for seed in range(args.runs):
        *_, m, _ = run(quiet=True, seed=seed)
        lead, foll = m["leader"], m["follower"]
        row = (seed, m["mission_time_s"], m["final_map_coverage"], lead["red_tiles"], foll["red_tiles"],
               lead["collisions"] + foll["collisions"], str(lead["crossed_border"] or foll["crossed_border"]),
               str(lead["finished_at_s"] is not None))
        rows.append(row)
        print("{:4d} | {:16.0f} | {:11.1%} | {:>8s}, {:<8s} | {:10d} | {:14} | {}".format(*row), flush=True)

    n = len(rows)
    print(f"\nmean mission time {sum(r[1] for r in rows) / n:.0f} s, "
          f"mean coverage {sum(r[2] for r in rows) / n:.1%}, "
          f"all tiles found in {sum(r[3] == '3/3' and r[4] == '3/3' for r in rows)}/{n} runs, "
          f"collision-free in {sum(r[5] == 0 for r in rows)}/{n} runs, "
          f"border respected in {sum(r[6] == 'False' for r in rows)}/{n} runs")


if __name__ == "__main__":
    main()
