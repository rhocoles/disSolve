import sqlite3
import time
from datetime import datetime
import socket
import subprocess
import sys
import argparse
import os

def format_timestamp_custom(unix_time):
    # Converts time.time() into 'day/month/year/hour/min/sec'
    return datetime.fromtimestamp(unix_time).strftime('%d/%m/%Y/%H/%M/%S')

def get_computer_name():
    # Grabs the hostname and ensures it is returned as a standard string
    computer_name = socket.gethostname()
    return str(computer_name)

def make_practice_BIG_db(name):
    DB_FILE = name+"_BIG.db"

    db = sqlite3.connect(DB_FILE)   #opens or creates a database call DB_FILE
    db.execute("PRAGMA foreign_keys=1")
    db.row_factory = sqlite3.Row
    cur = db.cursor()               #this is the curser which executes the sqlite command line scripts and returns the current position when iterating over results in the table

    q="""
    CREATE TABLE IF NOT EXISTS _experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rs REAL,                                                                -- solvent radius   (value in (0.02, 0.5)  3 dec places)
        eta REAL,                                                               -- eta              (value in (0, 0.495) 3 dec places
        status TEXT DEFAULT 'queued',                                           -- queued/in progress/completed, not copied/completed/failed
        startDate TEXT,                                                         -- start date
        completionDate TEXT,                                                    -- end date
        prefactors TEXT NOT NULL DEFAULT '',                                    -- [alpha_1, alpha_2, alpha_3, alpha_4]  (alpha_i in R)
        min_energy_normalised REAL,                                             -- (E - E0)/L with E0 computed using the embedded measures, L curve length
        min_energy_curve_id INTEGER,                                            -- curveID (local) of minimising example, in passing the data to the central database the id is updated with the new curveID
        size INTEGER,                                                           -- numer of parallel processes
        computerName TEXT,                                                      -- name of computer
        T_bot REAL,                                                             -- temperature lower bound
        T_top REAL,                                                             -- temperature upper bound
        temperature_description TEXT,                                           -- tempScan/geometric/linear
        number_of_rounds INTEGER,                                               -- number of rounds = exchanges attempted
        allgather_time REAL,                                                    -- seconds computed between rounds/attempted exchanges
        initial_curve_configs TEXT                                              -- list of curve id's e.g "[0,101,8,20000]" or "[1,23,47445]", "[0]" means basic example from pointFilaments.py, 0 BIG curveIDs, 1 local curveIDs
    )"""
    cur.execute(q)

    cur.execute('''
        INSERT INTO _experiments (
            rs,
            eta
        ) VALUES (?, ?)
    ''', (0.1, 0.05))
    db.commit()
    db.close()

    return None

def initialise_db(name, path_to_BIG_db=None, experimentID=None):

    DB_FILE = name+".db"
    if not os.path.exists(DB_FILE):
        print(f"local db {DB_FILE} not existing, making it now", file=sys.stderr)


    db = sqlite3.connect(DB_FILE)   #opens or creates a database call DB_FILE
    db.execute("PRAGMA foreign_keys=1")
    db.row_factory = sqlite3.Row
    cur = db.cursor()               #this is the curser which executes the sqlite command line scripts and returns the current position when iterating over results in the table

    q="""
    CREATE TABLE IF NOT EXISTS _experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rs REAL,                                                                -- solvent radius   (value in (0.02, 0.5)  3 dec places)
        eta REAL,                                                               -- eta              (value in (0, 0.495) 3 dec places
        startDate TEXT,                                                         -- start date
        completionDate TEXT,                                                    -- end date
        prefactors TEXT NOT NULL DEFAULT '',                                    -- [alpha_1, alpha_2, alpha_3, alpha_4]  (alpha_i in R)
        min_energy_normalised REAL,                                             -- (E - E0)/L with E0 computed using the embedded measures, L curve length
        min_energy_curve_id INTEGER,                                            -- curveID (local) of minimising example, in passing the data to the central database the id is updated with the new curveID
        size INTEGER,                                                           -- numer of parallel processes
        computerName TEXT,                                                      -- name of computer
        T_bot REAL,                                                             -- temperature lower bound
        T_top REAL,                                                             -- temperature upper bound
        temperature_description TEXT,                                           -- tempScan/geometric/linear
        number_of_rounds INTEGER,                                               -- number of rounds = exchanges attempted
        allgather_time REAL,                                                    -- seconds computed between rounds/attempted exchanges
        initial_curve_configs TEXT                                              -- list of curve id's e.g "[101, 8, 20000]" "[0]" means basic structure from pointFilaments is used
    )"""
    cur.execute(q)

    q="""
    CREATE TABLE IF NOT EXISTS _experiments_all_curves (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experimentID INTEGER,
        pointCoordinates TEXT,                                                   -- json string of the form "[[x, y, z], [x, y, z], ..., ]"
        frameNumber INTEGER,                                                     -- frame number
        rankNumber INTEGER,                                                      -- number of the parallel process
        time_in_sequence REAL,                                                   -- time in seconds relative to start time
        temperature REAL,                                                        -- temperature
        energy REAL,                                                             -- energy (normalised with respect to embedded energy and length)
        L REAL,                                                                  -- curve length (sum of edges)
        edgeLength REAL,                                                         -- average edge length between vertices
        numberOfBalls INTEGER,                                                   -- number of vertices
        reachConstraint REAL,                                                    -- lower bound of the curve's reach
        radiusGyration REAL,                                                     -- radius of gyration
        minLocalRadiusCurvature REAL,                                            -- minimum local radius of curvature
        selfDistance REAL,                                                       -- minimum distance between pair of double critcal points of distance function to curve
        FOREIGN KEY (experimentID) REFERENCES _experiments(id)                   -- experiment id
    )"""
    cur.execute(q)

    q="""
    CREATE TABLE IF NOT EXISTS _measures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        curveID INTEGER,
        experimentID INTEGER,
        inputSphereRadius REAL,                                                  -- this is the input radius with which the measures are computed
        V0 REAL,                                                                 -- embedded volume
        V REAL,                                                                  -- volume
        A0 REAL,                                                                 -- embedded surface area
        A REAL,                                                                  -- surface rea
        C0 REAL,                                                                 -- embedded integrated mean curvature of boundary
        C REAL,                                                                  -- integrated mean curvature of boundary
        X0 REAL,                                                                 -- embedded Euler characteristic
        X REAL,                                                                  -- Euler characteristic
        FOREIGN KEY (curveID) REFERENCES _experiments_all_curves(id),            -- curve example id
        FOREIGN KEY (experimentID) REFERENCES _experiments(id)                   -- experiment id
    )"""
    cur.execute(q)

    q="""
    CREATE TABLE IF NOT EXISTS _experiment_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experimentID INTEGER,
        round_nbr INTEGER,                                                      -- round number
        temperatures TEXT,                                                      -- T_0, T_1, ... Tn-1 where n= size
        prob TEXT,                                                              -- p_0, p_1, ... , pn-1 where n=size and p_i is the probability that binIndex i exchanges temperature with binIndex i+1 if i%2 == round_nbr%2
        acceptOrNot TEXT,                                                       -- a_0, a_1, a_2, ...a_n-1 where a_i = 1 means binIndex i exchanges temperature with binIndex i+1 and a_i = 0 means did not exchange iff i%2==round_nbr
        binIndices TEXT,                                                        -- 0, 2, n-1, 4 ... list of n integers in order rank: bin_index
        energy TEXT,                                                            -- E_0, E_1, ... En-1 where n=size E_i: energy rank i
        FOREIGN KEY (experimentID) REFERENCES _experiments(id)                  -- experiment id
    )"""
    cur.execute(q)

    q="""
    CREATE TABLE IF NOT EXISTS _experiment_stats_intra_rounds (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        curveID INTEGER,
        rank INTEGER,                                                           -- number of parallel chain
        it_no INTEGER,                                                          -- iteration number
        T REAL,                                                                 -- temperature
        prob REAL,                                                              -- probability of accepting new move
        deltaE REAL,                                                            -- energy difference between move
        accept  INTEGER,                                                        -- cumulative acceptance
        energy REAL,                                                            -- energy
        bin_index INTEGER,                                                      -- bin index
        time REAL,                                                              -- time stamp in seconds
        round_nbr INTEGER,                                                      -- round number
        it_no_intra_round INTEGER,                                              -- intra itereation number
        FOREIGN KEY (curveID) REFERENCES _experiments_all_curves(id)            -- curve id
    )"""
    cur.execute(q)

    startDate = format_timestamp_custom(time.time())
    compy = get_computer_name()

    if experimentID is None and not path_to_BIG_db: #perform experiment with new locally generated experimentID

        cur.execute('''INSERT INTO _experiments (startDate, computerName) VALUES (?, ?)''', (startDate, compy))
        experimentID = cur.lastrowid

    elif experimentID is None and path_to_BIG_db:  #perform experiment with new experimentID generated from row insertion into BIG_db

        BIG_db = sqlite3.connect(path_to_BIG_db)
        BIG_db.row_factory = sqlite3.Row
        BIG_cur = BIG_db.cursor()
        BIG_cur.execute('''INSERT INTO _experiments (startDate, computerName, status) VALUES (?, ?, ?)''', (startDate, compy, "in progress"))
        experimentID = BIG_cur.lastrowid
        BIG_db.commit()
        BIG_db.close()

        cur.execute('''INSERT INTO _experiments (id, startDate, computerName) VALUES (?, ?, ?)''', (experimentID, startDate, compy))

    elif experimentID is not None and not path_to_BIG_db: #add row with experimentID to local db if not existing otherwise generate new experimentID from row insertion to local db

        cur.execute("SELECT * FROM _experiments WHERE id = ?", (experimentID,))
        row = cur.fetchone()
        if row:
            print(f"Warning: An experiment with id {experimentID} already exists in local database. \nAssigning a new experimentID.")
            cur.execute('''INSERT INTO _experiments (startDate, computerName) VALUES (?, ?)''', (startDate, compy))
            experimentID = cur.lastrowid
        else:
            cur.execute('''INSERT INTO _experiments (id, startDate, computerName) VALUES (?, ?, ?)''', (experimentID, startDate, compy))

    else:#if experimentID exists in BIG_db, check experiment status, if queued update to inprogress and proceed, if not throw error. If experimentID does not exist in BIG_db, check if exists in local db. if not existing in local db insert new row into BIG db and proceed otherwise exit as above
        BIG_db = sqlite3.connect(path_to_BIG_db)
        BIG_db.row_factory = sqlite3.Row
        BIG_cur = BIG_db.cursor()

        BIG_cur.execute("SELECT * FROM _experiments WHERE id = ?", (experimentID,))
        row = BIG_cur.fetchone()
        if row is None:
            print(f"Warning: No experiment with id {experimentID} found in {path_to_BIG_db}")
            cur.execute("SELECT * FROM _experiments WHERE id = ?", (experimentID,))
            if cur.fetchone() is not None:
                BIG_db.close()
                sys.exit(f"An experiment with id {experimentID} already exists in local database {DB_FILE}. \nPlease review.")

            BIG_cur.execute('''INSERT INTO _experiments (id, startDate, computerName, status) VALUES (?, ?, ?, ?)''', (experimentID, startDate, compy, "in progress"))
            BIG_db.commit()
            BIG_db.close()

            cur.execute('''INSERT INTO _experiments (id, startDate, computerName) VALUES (?, ?, ?)''', (experimentID, startDate, compy))

        else:
            if row['status'] !='queued':
                BIG_db.close()
                sys.exit(f"Experiment with id {experimentID} is in progress or completed!")
            else:
                cur.execute("SELECT * FROM _experiments WHERE id = ?", (experimentID,))
                if cur.fetchone() is not None:
                    BIG_db.close()
                    sys.exit(f"An experiment with id {experimentID} already exists in local database {DB_FILE}. \nPlease review.")

                BIG_cur.execute("UPDATE _experiments SET status = 'in progress', startDate = ?, computerName = ? WHERE id = ?", (startDate, get_computer_name(), experimentID))
                BIG_db.commit()
                BIG_db.close()

                cur.execute('''INSERT INTO _experiments (id, rs, eta, startDate, size, computerName, T_bot, T_top, temperature_description, number_of_rounds, allgather_time, initial_curve_configs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    experimentID, row['rs'], row['eta'], startDate, row['size'], compy, row['T_bot'], row['T_top'],
                    row['temperature_description'], row['number_of_rounds'], row['allgather_time'], row['initial_curve_configs']
                ))
    db.commit()
    db.close()

    return experimentID

#If name is omitted, argparse prints a usage message to stderr and exits with code 2 automatically
#The two optional ones are --flag value style, so order doesn't matter and you can omit either or both independently
#Called like:
#python3 initialise_experiment_database.py circleTB --path_to_BIG_db circleTB_BIG.db --experimentID 5
#python3 initialise_experiment_database.py circleTB   # both optional args default to None

def do():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")  # required positional
    parser.add_argument("--path_to_BIG_db", default=None)
    parser.add_argument("--experimentID", type=int, default=None)
    args = parser.parse_args()

    experimentID = initialise_db(args.name, args.path_to_BIG_db, args.experimentID)
    print(experimentID)
    return None

do()
#make_practice_BIG_db("openChain50_dl0_25")
