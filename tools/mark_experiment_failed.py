import sqlite3
import sys
import os
from experiment_logging import BIG_DIR

#Called like: python3 mark_experiment_failed.py <structure> <experimentID>
#No confirmation prompt, no interactive input - meant to be called automatically from a
#detached screen session when mpirun exits nonzero, where no one is present to answer a prompt.
#Only touches BIG's status - local data is left alone for later inspection.

def mark_status_failed(name, experimentID):
    """Sets BIG's status to 'failed' for this experimentID. Does not touch local data."""
    path_to_BIG_db = os.path.join(BIG_DIR, name + "_BIG.db")
    if not os.path.exists(path_to_BIG_db):
        sys.exit(f"No BIG db found for {name} at {path_to_BIG_db}.")

    BIG_db = sqlite3.connect(path_to_BIG_db)
    cur = BIG_db.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='_experiments'")
    if cur.fetchone() is None:
        BIG_db.close()
        sys.exit(f"BIG db for {name} has no _experiments table. You likely need to delete ghost _BIG.db")

    BIG_db.execute("UPDATE _experiments SET status = 'failed' WHERE id = ?", (experimentID,))
    BIG_db.commit()
    BIG_db.close()
    print(f"Marked experiment {experimentID} as failed in BIG.")
    return None


if __name__ == "__main__":
    mark_status_failed(sys.argv[1], int(sys.argv[2]))
