import sqlite3
import json
import os
import shutil
import argparse
import glob
import sys

BIG_DIR = "/Users/harmon/Results/BIG_DIR/"

def get_local_experiment(name, experimentID):
    DB_FILE = name + ".db"
    db = sqlite3.connect(DB_FILE)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    cur.execute("SELECT * FROM _experiments WHERE id = ?", (experimentID,))
    row = cur.fetchone()
    db.close()
    return row

def get_BIG_experiment(name, experimentID):
    path_to_BIG_db = os.path.join(BIG_DIR, name + "_BIG.db")
    db = sqlite3.connect(path_to_BIG_db)
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    cur.execute("SELECT * FROM _experiments WHERE id = ?", (experimentID,))
    row = cur.fetchone()
    db.close()
    return row


def update_BIG_metadata(name, experimentID, row, status, min_energy_curve_id=None):
    # experimentID may not have an existing BIG row yet (e.g. a purely local-only run)
    # method update BIG with metadata or inserts row if run was local
    # status in progress/completed/completed, not copied/failed
    # min_energy_curve_id must be a BIG-side curveID (or None), never a local one

    path_to_BIG_db = os.path.join(BIG_DIR, name + "_BIG.db")
    BIG_db = sqlite3.connect(path_to_BIG_db)
    cur = BIG_db.cursor()

    cur.execute("SELECT id FROM _experiments WHERE id = ?", (experimentID,))
    exists = cur.fetchone()

    if exists:
        cur.execute('''
            UPDATE _experiments SET
                rs = ?, eta = ?, prefactors = ?, startDate = ?, completionDate = ?,
                min_energy_normalised = ?, min_energy_curve_id = ?, size = ?, computerName = ?,
                T_bot = ?, T_top = ?, temperature_description = ?, number_of_rounds = ?,
                allgather_time = ?, initial_curve_configs = ?, status = ?
            WHERE id = ?
        ''', (
            row['rs'], row['eta'], row['prefactors'], row['startDate'], row['completionDate'],
            row['min_energy_normalised'], min_energy_curve_id, row['size'], row['computerName'],
            row['T_bot'], row['T_top'], row['temperature_description'], row['number_of_rounds'],
            row['allgather_time'], row['initial_curve_configs'], status,
            experimentID
        ))
    else:
        cur.execute('''
            INSERT INTO _experiments (
                id, rs, eta, prefactors, startDate, completionDate,
                min_energy_normalised, min_energy_curve_id, size, computerName,
                T_bot, T_top, temperature_description, number_of_rounds,
                allgather_time, initial_curve_configs, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experimentID, row['rs'], row['eta'], row['prefactors'], row['startDate'], row['completionDate'],
            row['min_energy_normalised'], min_energy_curve_id, row['size'], row['computerName'],
            row['T_bot'], row['T_top'], row['temperature_description'], row['number_of_rounds'],
            row['allgather_time'], row['initial_curve_configs'], status
        ))

    BIG_db.commit()
    BIG_db.close()
    return None


def add_tables_to_BIG_db(name):
    path_to_BIG_db = os.path.join(BIG_DIR, name + "_BIG.db")
    BIG_db = sqlite3.connect(path_to_BIG_db)
    cur = BIG_db.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS _experiments_all_curves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experimentID INTEGER,
            pointCoordinates TEXT,
            frameNumber INTEGER,
            rankNumber INTEGER,
            time_in_sequence REAL,
            temperature REAL,
            energy REAL,
            L REAL,
            edgeLength REAL,
            numberOfBalls INTEGER,
            reachConstraint REAL,
            radiusGyration REAL,
            minLocalRadiusCurvature REAL,
            selfDistance REAL,
            FOREIGN KEY (experimentID) REFERENCES _experiments(id)
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS _measures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            curveID INTEGER,
            experimentID INTEGER,
            inputSphereRadius REAL,
            V0 REAL,
            V REAL,
            A0 REAL,
            A REAL,
            C0 REAL,
            C REAL,
            X0 REAL,
            X REAL,
            FOREIGN KEY (curveID) REFERENCES _experiments_all_curves(id),
            FOREIGN KEY (experimentID) REFERENCES _experiments(id)
        )
    ''')
    BIG_db.commit()
    BIG_db.close()
    return None

def delete_local_experiment( name, experimentID):
    if os.path.exists(name+".db"):
        db = sqlite3.connect(name+".db")
        cur = db.cursor()
        cur.execute("DELETE FROM _measures WHERE experimentID = ?", (experimentID,))
        cur.execute("DELETE FROM _experiments_all_curves WHERE experimentID = ?", (experimentID,))
        cur.execute("DELETE FROM _experiments WHERE id = ?", (experimentID,))
        db.commit()
        db.close()
    else:
        print(f"directory constains no {name} database.")
    return None

def return_centered_polyFile_and_move_to_BIG_poly_dir(name, local_poly_path, curveData, BIG_curveID):
    """ Function updates the point positions as given in curve_local_curveID.poly with coordinates as in curveData (centered at the origin)
        new polyFiles is saved with BIGcurveID in name_polyFiles in BIG_DIR"""

    with open(local_poly_path, 'r') as f:
        lines = f.read().splitlines()

    strand_lines = lines[lines.index('POLYS') + 1: lines.index('END')]
    is_closed_per_strand = []
    for line in strand_lines:
        indices = list(map(int, line.split(': ')[1].split()))
        is_closed_per_strand.append(indices[0] == indices[-1])

    if len(is_closed_per_strand) != len(curveData):
        raise ValueError(
            f"strand count mismatch for curve {BIG_curveID}: "
            f"{len(is_closed_per_strand)} in local poly file vs {len(curveData)} in pointCoordinates"
        )

    os.makedirs(BIG_DIR+name+"_polyFiles", exist_ok=True)
    BIG_poly_path = BIG_DIR+name+"_polyFiles/curve_"+str(BIG_curveID)+".poly"

    with open(BIG_poly_path, "w") as f:
        f.write("POINTS\n")
        point_index = 1
        strand_indices = []
        for strand in curveData:
            indices = []
            for point in strand:
                x, y, z = point
                f.write(f"{point_index}: {x} {y} {z}\n")
                indices.append(point_index)
                point_index += 1
            strand_indices.append(indices)

        f.write("POLYS\n")
        for i, indices in enumerate(strand_indices):
            if is_closed_per_strand[i]:
                indices = indices + [indices[0]]
            f.write(f"{i+1}: " + " ".join(map(str, indices)) + "\n")

        f.write("END\n")

    return None

def log_missing_polyFile(name, experimentID, curveID):
    with open(BIG_DIR + name + f"_polyFile_{experimentID}_missing_curveIDs", "a") as f:
        f.write(f"{curveID}\n")
    return None

def delete_local_polyFile_archive(experimentID):
    extract_dir = f"polyFiles_{experimentID}"
    zip_path = f"polyFiles_{experimentID}.zip"

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    return None
    
def move_data_to_BIG(name, experimentID, batch_size=100):
    #check if experimentID determines an experiment in local directory
    row = get_local_experiment(name, experimentID)
    if row is None:
        print(f"No local experiment with id {experimentID}")
        return None
    local_min_curveID = row["min_energy_curve_id"]

    #unzip the polyFiles_{experimentID} folder and log error if missing
    zip_path = f"polyFiles_{experimentID}.zip"
    extract_dir = f"polyFiles_{experimentID}"

    if os.path.exists(zip_path):
        shutil.unpack_archive(zip_path, extract_dir)
        status = "completed"
    else:
        print(f"polyFiles archive not found: {zip_path} - all poly files for this experiment are missing")
        status = "completed, missing polyFiles"
        extract_dir = None

    add_tables_to_BIG_db(name)

    DB_FILE = name + ".db"
    local_db = sqlite3.connect(DB_FILE)
    local_db.row_factory = sqlite3.Row
    local_cur = local_db.cursor()

    path_to_BIG_db = os.path.join(BIG_DIR, name + "_BIG.db")
    BIG_db = sqlite3.connect(path_to_BIG_db)
    BIG_db.row_factory = sqlite3.Row
    BIG_cur = BIG_db.cursor()

    local_cur.execute("SELECT id FROM _experiments_all_curves WHERE experimentID = ?", (experimentID,))
    curveIDs = [r['id'] for r in local_cur.fetchall()]

    BIG_min_curveID = None

    # process in small batches: insert a batch into BIG and commit it, then
    # only delete that same batch locally and commit that. A crash mid-way
    # leaves at most one batch temporarily duplicated - never deleted locally
    # without having safely landed in BIG first.
    for i in range(0, len(curveIDs), batch_size):
        batch = curveIDs[i:i + batch_size] #batch is a list of curveIDs (e.g. [101, 102, 103])

        for curveID in batch:
            local_cur.execute("SELECT * FROM _experiments_all_curves WHERE id = ?", (curveID,))
            curve = local_cur.fetchone()

            local_cur.execute("SELECT * FROM _measures WHERE curveID = ?", (curveID,))
            measure = local_cur.fetchone()

            BIG_cur.execute('''
                INSERT INTO _experiments_all_curves (
                    experimentID, pointCoordinates, frameNumber, rankNumber, time_in_sequence,
                    temperature, energy, L, edgeLength, numberOfBalls, reachConstraint,
                    radiusGyration, minLocalRadiusCurvature, selfDistance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experimentID, curve['pointCoordinates'], curve['frameNumber'], curve['rankNumber'],
                curve['time_in_sequence'], curve['temperature'], curve['energy'], curve['L'], curve['edgeLength'],
                curve['numberOfBalls'], curve['reachConstraint'], curve['radiusGyration'],
                curve['minLocalRadiusCurvature'], curve['selfDistance']
            ))
            BIG_curveID = BIG_cur.lastrowid

            if curveID == local_min_curveID:
                BIG_min_curveID = BIG_curveID

            if measure is not None:
                BIG_cur.execute('''
                    INSERT INTO _measures (
                        curveID, experimentID, inputSphereRadius, V0, V, A0, A, C0, C, X0, X
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    BIG_curveID, experimentID, measure['inputSphereRadius'], measure['V0'], measure['V'],
                    measure['A0'], measure['A'], measure['C0'], measure['C'], measure['X0'], measure['X']
                ))

            if extract_dir is not None:
                local_poly_path = f"polyFiles_{experimentID}/test_{curve['rankNumber']}_{curve['frameNumber']}.poly"
                if os.path.exists(local_poly_path):
                    return_centered_polyFile_and_move_to_BIG_poly_dir(name, local_poly_path, json.loads(curve['pointCoordinates']), BIG_curveID)
                else:
                    log_missing_polyFile(name, experimentID, BIG_curveID)

        BIG_db.commit()

        placeholders = ','.join('?' * len(batch))
        local_cur.execute(f"DELETE FROM _measures WHERE curveID IN ({placeholders})", batch)
        local_cur.execute(f"DELETE FROM _experiments_all_curves WHERE id IN ({placeholders})", batch)
        local_db.commit()

        print(f"copied {min(i + batch_size, len(curveIDs))}/{len(curveIDs)} curves")

    local_cur.execute("DELETE FROM _experiments WHERE id = ?", (experimentID,))
    local_db.commit()

    BIG_db.close()
    local_db.close()
    
    #delete the polyFiles but not the zip file
    if extract_dir is not None:
        shutil.rmtree(extract_dir)
    
    update_BIG_metadata(name, experimentID, row, status, min_energy_curve_id=BIG_min_curveID)
    print(f"Moved experiment {experimentID} to BIG and cleared local copy.")

    return None

def mark_as_completed_but_do_not_move(name, experimentID):
    row = get_local_experiment(name, experimentID)
    if row is None:
        print(f"No local experiment with id {experimentID}")
        return None

    update_BIG_metadata(name, experimentID, row, "completed, not copied")
    print(f"Marked experiment {experimentID} as completed in BIG. Local data untouched.")
    return None


def delete_data_maintain_record(name, experimentID):

    row = get_local_experiment(name, experimentID)
    if row is None:
        print(f"No local experiment with id {experimentID}")
        return None

    update_BIG_metadata(name, experimentID, row, "failed")
    delete_local_experiment(name, experimentID)
    delete_local_polyFile_archive(experimentID)
    
    print(f"Marked experiment {experimentID} as failed in BIG and deleted local data.")
    return None

def delete_data_and_record(name, experimentID):

    delete_local_experiment(name, experimentID)
    delete_local_polyFile_archive(experimentID)

    path_to_BIG_db = os.path.join(BIG_DIR, name + "_BIG.db")
    BIG_db = sqlite3.connect(path_to_BIG_db)
    cur = BIG_db.cursor()
    cur.execute("DELETE FROM _experiments WHERE id = ?", (experimentID,))
    BIG_deleted = cur.rowcount > 0
    BIG_db.commit()
    BIG_db.close()

    if BIG_deleted:
        print(f"Deleted experiment {experimentID} from both local db, polyFiles and BIG.")
    else:
        print(f"Deleted experiment {experimentID} from local db and polyFiles. No matching record found in BIG.")
    return None

def get_current_local_experiment_update(name, experimentID):

    local_row = get_local_experiment(name, experimentID)
    BIG_row = get_BIG_experiment(name, experimentID)

    print(f"--- Experiment {experimentID} ---")

    if local_row is None:
        print("Local db: no entry.")
    else:
        local_status = "completed (evaluated)" if local_row['completionDate'] else "in progress / not yet evaluated"
        date_label = "completed" if local_row['completionDate'] else "started"
        date_str = local_row['completionDate'] if local_row['completionDate'] else local_row['startDate']
        print(f"Local db: rs={local_row['rs']}, eta={local_row['eta']}, status={local_status}, {date_label} {date_str}")

    if BIG_row is None:
        print("BIG db: no entry.")
    else:
        date_label = "completed" if BIG_row['completionDate'] else "started"
        date_str = BIG_row['completionDate'] if BIG_row['completionDate'] else BIG_row['startDate']
        print(f"BIG db: rs={BIG_row['rs']}, eta={BIG_row['eta']}, status={BIG_row['status']}, {date_label} {date_str}")

    return local_row, BIG_row

def get_overview_of_directory():
    local_db_files = sorted(f for f in glob.glob("*.db") if not f.endswith("_BIG.db"))
    if not local_db_files:
        print("No local .db files found in current directory.")
        return

    for db_file in local_db_files:
        name = db_file[:-3]
        path_to_BIG_db = os.path.join(BIG_DIR, name + "_BIG.db")
        BIG_exists = os.path.exists(path_to_BIG_db)
        print(f"\n=== {name} ===  (BIG db: {'found' if BIG_exists else 'not found'})")

        local_db = sqlite3.connect(db_file)
        local_db.row_factory = sqlite3.Row
        total = local_db.execute("SELECT COUNT(*) AS n FROM _experiments").fetchone()["n"]
        recent_rows = local_db.execute(
            "SELECT id, startDate, completionDate FROM _experiments ORDER BY startDate DESC LIMIT 5"
        ).fetchall()
        local_db.close()

        print(f"local: {total} experiment(s) total, showing {len(recent_rows)} most recent")

        ids_in_BIG = set()
        if BIG_exists:
            recent_ids = [row["id"] for row in recent_rows]
            if recent_ids:
                placeholders = ",".join("?" * len(recent_ids))
                BIG_db = sqlite3.connect(path_to_BIG_db)
                ids_in_BIG = {
                    row[0] for row in BIG_db.execute(
                        f"SELECT id FROM _experiments WHERE id IN ({placeholders})", recent_ids
                    ).fetchall()
                }
                BIG_db.close()

        for row in recent_rows:
            line = f"  experimentID={row['id']}  startDate={row['startDate']}   completionDate={row['completionDate']}"
            if BIG_exists:
                line += f"  in BIG: {'yes' if row['id'] in ids_in_BIG else 'no'}"
            print(line)

        if recent_rows:
            example_id = recent_rows[0]["id"]
            print(f"  for full detail: python3 experiment_logging.py {name} {example_id}")
    return None

def do():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default=None)
    parser.add_argument("experimentID", nargs="?", type=int, default=None)
    parser.add_argument("--action", choices=["move", "mark", "failed", "delete-all"])
    args = parser.parse_args()

    if args.name is None:
        get_overview_of_directory()
        return None

    if args.experimentID is None:
        sys.exit(f"Give an experimentID to look at, e.g. python3 experiment_logging.py {args.name} <experimentID>")
        return None

    get_current_local_experiment_update(args.name, args.experimentID)

    if args.action is None:
        
        print("\nNo action given. Options:")
        print("  --action move                move local db data to BIG, delete local, keep polyFile zip, add copy of curve in BIG polyFile directory")
        print("  --action mark                update status as completed in BIG, but do not copy the data from local db, keep polyFiles")
        print("  --action failed              update status as failed in BIG, delete local data from local db and delete polyFiles")
        print("  --action delete-all          delete all experimentID data from both local db and BIG db, delete polyFiles")
        return

    confirm = input(f"\nProceed with '{args.action}' for experiment {args.experimentID}? [y/N] ")
    if confirm.lower() != "y":
        print("Cancelled.")
        return

    if args.action == "move":
        move_data_to_BIG(args.name, args.experimentID)
    elif args.action == "mark":
        mark_as_completed_but_do_not_move(args.name, args.experimentID)
    elif args.action == "failed":
        delete_data_maintain_record(args.name, args.experimentID)
    elif args.action == "delete-all":
        delete_data_and_record(args.name, args.experimentID)
    
    return None

# Only runs do() when this file is executed directly (python3 experiment_logging.py ...).
# If another script does `import experiment_logging`, __name__ is "experiment_logging" instead
# of "__main__", so do() (and its argparse parsing) is skipped.
if __name__ == "__main__":
    do()

