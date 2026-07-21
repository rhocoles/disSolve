import sqlite3
import argparse
import os
import sys
import numpy as np
from datetime import datetime
from experiment_logging import BIG_DIR #Python automatically adds the running script's own directory to its import search path, so it'll find experiment_logging.py in tools/ not the experiment_logging.py copied to the local experiment directory. This is ok


def get_db_path(name, source):
    db_path = os.path.join(BIG_DIR, name + "_BIG.db") if source == "BIG" else name + ".db"
    if not os.path.exists(db_path):
        sys.exit(f"Database not found: {db_path}")
    return db_path


def has_column(db, table, column):
    return column in [row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()]


def compute_and_cache_min_energy(db, experimentID):
    """energy = dot(prefactors, [V-V0, A-A0, C-C0, X-X0]) / L, computed for every curve in this
       experiment, cached into _experiments_all_curves.energy, then the minimum is cached into
       _experiments.min_energy_curve_id / min_energy_normalised."""
    prefactors_row = db.execute(
        "SELECT prefactors FROM _experiments WHERE id = ?", (experimentID,)
    ).fetchone()
    if prefactors_row is None or not prefactors_row["prefactors"] or not prefactors_row["prefactors"].strip():
        sys.exit(f"No prefactors recorded for experimentID={experimentID} - cannot compute energy")

    c = np.array([float(x) for x in prefactors_row["prefactors"].split()])

    rows = db.execute('''
        SELECT curves.id AS curveID, curves.L AS L,
               m.V0 AS V0, m.V AS V, m.A0 AS A0, m.A AS A,
               m.C0 AS C0, m.C AS C, m.X0 AS X0, m.X AS X
        FROM _experiments_all_curves AS curves
        JOIN _measures AS m ON m.curveID = curves.id
        WHERE curves.experimentID = ?
    ''', (experimentID,)).fetchall()

    if not rows:
        return

    best_id, best_energy = None, None
    updates = []
    for row in rows:
        m = np.array([row["V"] - row["V0"], row["A"] - row["A0"], row["C"] - row["C0"], row["X"] - row["X0"]])
        energy = float(np.dot(c, m) / row["L"])
        updates.append((energy, row["curveID"]))
        if best_energy is None or energy < best_energy:
            best_id, best_energy = row["curveID"], energy

    # loops over updates, built up during the per-row loop above. db.executemany: it's a sqlite3
    # method that runs the same SQL statement once per item in a sequence, instead of you writing
    # a Python loop calling .execute() each time
    db.executemany("UPDATE _experiments_all_curves SET energy = ? WHERE id = ?", updates)
    db.execute(
        "UPDATE _experiments SET min_energy_curve_id = ?, min_energy_normalised = ? WHERE id = ?",
        (best_id, best_energy, experimentID)
    )
    return None


def get_min_energy_curveID(name, source, rs, eta):
    """Returns the curveID of minimal energy among completed experiments matching rs and eta.
       Computes and caches min_energy_curve_id/min_energy_normalised for any matching experiment
       missing it. Skips BIARC-type experiments if curveDataStructure exists on this db."""
    db_path = get_db_path(name, source)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    biarc_filter = ""
    if has_column(db, "_experiments", "curveDataStructure"):
        biarc_filter = "AND (curveDataStructure IS NULL OR curveDataStructure != 'Biarcs')"

    status_filter = ""
    if has_column(db, "_experiments", "status"):
        status_filter = "AND status = 'completed'"

    candidates = db.execute(f'''
        SELECT id, min_energy_curve_id FROM _experiments
        WHERE round(rs, 3) = ? AND round(eta, 3) = ? {biarc_filter} {status_filter}
    ''', (round(rs, 3), round(eta, 3))).fetchall()

    if not candidates:
        db.close()
        sys.exit(f"No experiment found with rs={rs}, eta={eta} in {db_path}")

    for c in candidates:
        if c["min_energy_curve_id"] is None:
            compute_and_cache_min_energy(db, c["id"])
    db.commit()

    row = db.execute(f'''
        SELECT id, min_energy_curve_id, min_energy_normalised
        FROM _experiments
        WHERE round(rs, 3) = ? AND round(eta, 3) = ? AND min_energy_curve_id IS NOT NULL {biarc_filter} {status_filter}
        ORDER BY min_energy_normalised ASC
        LIMIT 1
    ''', (round(rs, 3), round(eta, 3))).fetchone()
    db.close()

    if row is None:
        sys.exit(f"No completed experiment with computable energy found for rs={rs}, eta={eta} in {db_path}")

    print(f"inital curve is lowest energy curve from {source} with rs={rs}, eta={eta}\n", file=sys.stderr)
    print(f"curveID: {row['min_energy_curve_id']} energy: {row['min_energy_normalised']} of experiment:{row['id']}\n", file=sys.stderr)

    return row["min_energy_curve_id"]


def get_most_recent_experimentID(name, source, rs, eta):
    """Returns the experimentID with the most recent completionDate matching rs and eta."""
    db_path = get_db_path(name, source)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    if has_column(db, "_experiments", "status"):
        status_filter = "AND status = 'completed'"
    else:
        status_filter = "AND completionDate IS NOT NULL"

    #if status is completed, not copied then curveIDs can not be given from BIG, they may still exist in the local db
    rows = db.execute(f'''
        SELECT id, completionDate FROM _experiments
        WHERE round(rs, 3) = ? AND round(eta, 3) = ? {status_filter}
    ''', (round(rs, 3), round(eta, 3))).fetchall()
    db.close()
    if not rows:
        sys.exit(f"No completed experiment found with rs={rs}, eta={eta} in {db_path}")

    with_dates = [r for r in rows if r["completionDate"]]
    if with_dates:
        latest = max(with_dates, key=lambda r: datetime.strptime(r["completionDate"], "%d/%m/%Y/%H/%M/%S"))
    else:
        latest = max(rows, key=lambda r: r["id"])
    
    return latest["id"]


def get_last_frame_curveIDs_per_rank(name, source, experimentID):
    """Returns a list of curveIDs, one per rank: the last (highest frameNumber) curve for that rank."""
    db_path = get_db_path(name, source)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute('''
        SELECT id, rankNumber, energy
        FROM _experiments_all_curves
        WHERE experimentID = ?
        AND (rankNumber, frameNumber) IN (
            SELECT rankNumber, MAX(frameNumber)
            FROM _experiments_all_curves
            WHERE experimentID = ?
            GROUP BY rankNumber
        )
        ORDER BY rankNumber
    ''', (experimentID, experimentID)).fetchall()
    row_check = db.execute('''
        SELECT id, rs, eta
        FROM _experiments
        WHERE id = ?
    ''', (experimentID, )).fetchone()
    db.close()
    if not rows:
        sys.exit(f"No curves found for experimentID={experimentID} in {db_path}")

    print(f"inital curves set as last frame from {source} for experiment {experimentID}", file=sys.stderr)
    print(f"experiment corresponds to rs={row_check['rs']} and eta={row_check['eta']}\n", file=sys.stderr)
    for row in rows:
        print(f"curveID: {row['id']} energy: {row['energy']} rankNumber:{row['rankNumber']}\n", file=sys.stderr)

    return [row["id"] for row in rows]


def get_min_energy_curveIDs_per_rank(name, source, experimentID):
    """Returns a list of curveIDs, one per rank: the minimal-energy curve for that rank."""
    db_path = get_db_path(name, source)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    rows = db.execute('''
        SELECT id, rankNumber, energy
        FROM _experiments_all_curves
        WHERE experimentID = ?
        AND (rankNumber, energy) IN (
            SELECT rankNumber, MIN(energy)
            FROM _experiments_all_curves
            WHERE experimentID = ?
            GROUP BY rankNumber
        )
        ORDER BY rankNumber
    ''', (experimentID, experimentID)).fetchall()
    row_check = db.execute('''
        SELECT id, rs, eta
        FROM _experiments
        WHERE id = ?
    ''', (experimentID, )).fetchone()
    db.close()
    if not rows:
        sys.exit(f"No curves found for experimentID={experimentID} in {db_path}")
    print(f"inital curves set as min energy per rank from {source} for experiment {experimentID}", file=sys.stderr)
    print(f"experiment corresponds to rs={row_check['rs']} and eta={row_check['eta']}\n", file=sys.stderr)
    for row in rows:
        print(f"curveID: {row['id']} energy: {row['energy']} rankNumber:{row['rankNumber']}\n", file=sys.stderr)
    return [row["id"] for row in rows]


#Called like:
#python3 select_initial_curve_configs.py circleTB --source BIG --option min-energy --rs 0.1 --eta 0.05
#python3 select_initial_curve_configs.py circleTB --source BIG --option last-frame-per-rank --experimentID 12
#python3 select_initial_curve_configs.py circleTB --source local --option min-energy-per-rank --rs 0.1 --eta 0.05
#   (--experimentID omitted -> resolves to the most recent experiment matching --rs/--eta)
#
#min-energy prints a single curveID; last-frame-per-rank/min-energy-per-rank print a
#comma-separated list, one curveID per rank, e.g. "104,205,301"

def do():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--source", required=True, choices=["local", "BIG"])
    parser.add_argument("--option", required=True,
                         choices=["min-energy", "last-frame-per-rank", "min-energy-per-rank"])
    parser.add_argument("--rs", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--experimentID", type=int)
    args = parser.parse_args()

    if args.source == "local" and not os.path.exists(args.name + ".db"):
        print(f"Error: local db {args.name}.db not found in current directory", file=sys.stderr)
        sys.exit(1)

    if args.option == "min-energy":
        print(get_min_energy_curveID(args.name, args.source, args.rs, args.eta))
        return

    experimentID = args.experimentID
    if experimentID is None:
        experimentID = get_most_recent_experimentID(args.name, args.source, args.rs, args.eta)

    if args.option == "last-frame-per-rank":
        ids = get_last_frame_curveIDs_per_rank(args.name, args.source, experimentID)
    elif args.option == "min-energy-per-rank":
        ids = get_min_energy_curveIDs_per_rank(args.name, args.source, experimentID)
    print(",".join(str(i) for i in ids))


if __name__ == "__main__":
   do()
