import sqlite3
import os
import time
from datetime import datetime
import socket
import papermill as pm
import subprocess
from email.message import EmailMessage


def format_timestamp_custom(unix_time):
    # Converts time.time() into 'day/month/year/hour/min/sec'
    return datetime.fromtimestamp(unix_time).strftime('%d/%m/%Y/%H/%M/%S')

def get_computer_name():
    # Grabs the hostname and ensures it is returned as a standard string
    computer_name = socket.gethostname()
    return str(computer_name)

def initialise_db(name):

    DB_FILE = name+".db"
    
    db = sqlite3.connect(DB_FILE)   #opens or creates a database call DB_FILE
    db.execute("PRAGMA foreign_keys=1") 
    db.row_factory = sqlite3.Row
    cur = db.cursor()               #this is the curser which executes the sqlite command line scripts and returns the current position when iterating over results in the table
    
    q="""
    CREATE TABLE IF NOT EXISTS _experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rs TEXT NOT NULL DEFAULT '',                                            -- solvent radius   (value in (0.02, 0.5)  3 dec places)
        eta TEXT NOT NULL DEFAULT '',                                           -- eta              (value in (0, 0.495) 3 dec places 
        prefactors TEXT NOT NULL DEFAULT '',                                    -- [alpha_1, alpha_2, alpha_3, alpha_4]  (alpha_i in R)
        prefactors_normalised_leading_term_is_one TEXT NOT NULL DEFAULT '',     -- [1, alpha_2/alpha_1, alpha_3/alpha_1, alpha_4/alpha_1] 
        minSolvationFreeEnergy REAL,                                            -- Fsol computed using prefactors 
        minEnergy REAL,                                                         -- energy computed using prefactors_normalised_leading_term_is_one            
        minEnergy_normalised REAL,                                              -- E - E0 with E0 computed using the embedded measures and both energies computed using the prefactors_normalised_leading_term_is_one
        min_energy_computed TEXT NOT NULL DEFAULT '',                           -- string list of the form (E - E0 etc, curve_id) so if you compute the energy anew of a given curve example, this in information is added here in this form. I would extend this to be a list of such tuples and organise by date, but lets see how useful this is
        startDate TEXT,                                                         -- start date
        completionDate TEXT,                                                    -- end date
        size INTEGER,                                                           -- numer of parallel processes
        computerName TEXT,                                                      -- name of computer
        temperature TEXT,                                                       -- (T_bot, T_top, description)
        number_of_rounds INTEGER,                                               -- number of rounds = exchanges attempted
        allgather_time REAL                                                     -- seconds computed between rounds/attempted exchanges
    )"""
    cur.execute(q)
    
    q="""
    CREATE TABLE IF NOT EXISTS _experiments_all_curves (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experimentID INTEGER,                                                   
        pointCoordinates TEXT,                                                   -- string of the form [[x, y, z], [x, y, z], ..., ]
        frameNumber INTEGER,                                                     -- frame number    
        rankNumber INTEGER,                                                      -- number of the parallel process
        temp REAL,                                                               -- temperature
        energy REAL,                                                             -- energy (normalised with respect to embedded energy and length)
        time_in_sequence REAL,                                                   -- time in sequence (references start time of the experiment)
        FOREIGN KEY (experimentID) REFERENCES _experiments(id)                   -- experiment id
    )"""
    cur.execute(q)
    
    q="""
    CREATE TABLE IF NOT EXISTS _measures (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        curveID INTEGER,
        inputSphereRadius REAL,                                                  -- this is the input radius with which the measures are computed
        V0 REAL,                                                                 -- embedded volume
        V REAL,                                                                  -- volume 
        A0 REAL,                                                                 -- embedded surface area
        A REAL,                                                                  -- surface rea 
        C0 REAL,                                                                 -- embedded integrated mean curvature of boundary
        C REAL,                                                                  -- integrated mean curvature of boundary
        X0 REAL,                                                                 -- embedded Euler characteristic
        X REAL,                                                                  -- Euler characteristic 
        L REAL,                                                                  -- curve length (sum of edges)
        edgeLength REAL,                                                         -- average edge length between vertices
        numberOfBalls INTEGER,                                                   -- number of vertices
        radiusGyration REAL,                                                     -- radius of gyration (for sorting data, needs to be computed and updated in a separate script)
        reachOfClosedComplement REAL,                                            -- reach of closed complement of ball union for given inputSphereRadius (for sorting data, needs to be computed and updated in a separate script)   
        FOREIGN KEY (curveID) REFERENCES _experiments_all_curves(id)             -- curve example id
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
        FOREIGN KEY (curveID) REFERENCES _experiments_all_curves(id)            -- curve id
    )"""
    cur.execute(q)

    db.commit()
    db.close()

    return None


def load_experiment_data():

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(os.path.join(data_dir, "experimentData.txt")):
        raise FileNotFoundError(f"Could not find experimentData.txt")
        
    with open(os.path.join(data_dir, "experimentData.txt")) as f:
        lines = f.read().splitlines()
        line = lines[-1].split(' ')
    #line.pop()

    # Parse variables exactly as you laid out
    size = int(float(line[0]))
    overlapRatio = float(line[1])
    eta = float(line[2])
    R = float(line[3])
    Rs = float(line[4])
    inputSphereRadius = float(line[5])
    
    prefactors = [float(line[6]), float(line[7]), float(line[8]), float(line[9])]
    sphereCount = int(float(line[10]))
    edgeLength = float(line[11])
    T_bot = float(line[12])
    T_top = float(line[13])
    numberOfRounds = int(float(line[14]))
    allgather_time = int(float(line[15]))
    geometric = bool(int(float(line[19])))
    if geometric:
        description = 'geometric'
    else:
        description = 'linear'
    temperature_str = '('+str(T_bot)+','+str(T_top)+', '+description+')'
    
    db_name = str(line[16])#Structure
    alpha = str(line[17])
    start = format_timestamp_custom(float(line[18]))

    end = format_timestamp_custom(time.time())
    comp_name = get_computer_name()

    if not os.path.exists(os.path.join(data_dir, "temperatures.txt")):
        raise FileNotFoundError(f"Could not find temperatures.txt")

    with open(os.path.join(data_dir, "temperatures.txt"), "r") as f:
        temperatures_str = f.read().strip()
    temperatures = str([float(t) for t in temperatures_str.split(" ") if t])

    initialise_db(db_name)
    DB_FILE = f"{db_name}.db"
    db = sqlite3.connect(DB_FILE)
    db.execute("PRAGMA foreign_keys=1")
    cur = db.cursor()

    cur.execute('''
        INSERT INTO _experiments (
            rs, 
            eta, 
            prefactors, 
            startDate,
            completionDate,
            size,
            computerName,
            temperature,
            number_of_rounds,
            allgather_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        f"{overlapRatio:.3f}",
        f"{eta:.3f}",
        str(prefactors),
        start,
        end,
        size,
        comp_name,
        temperature_str, 
        numberOfRounds, 
        allgather_time
    ))

    experiment_id = cur.lastrowid

    db.commit()
    db.close()

    return db_name, experiment_id

def read_point_coordinates_from_polyFile(file_name):
    if not os.path.exists(file_name):
        print(f"Warning: Poly file not found at {file_name}")
        return ""
        
    coordinates_list = ""
    reading_points = False
    
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()

            if line == "POINTS":
                reading_points = True
                continue
            elif line == "POLYS":
                break
                
            if reading_points:
                # Example line -> "1: -14.77499 -0.41406 -11.20048 c(0.055,0.471,0.8,1.0)"
                # Split at 'c(' to isolate the coordinates from the color values
                coord_part = line.split("c(")[0]
                
                # Split at ':' to drop the line number prefix
                xyz = coord_part.split(":")[1].strip()+", "
                
                coordinates_list+=xyz
                    
    return str(coordinates_list)

def load_curve_data(experiment_id, db_name):
    # Mac/Linux safe paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    data_dir = os.path.join(base_dir, "data")
    poly_dir = os.path.join(base_dir, "polyFiles")

    if not os.path.exists(os.path.join(data_dir, "experimentData.txt")):
        raise FileNotFoundError(f"Could not find experimentData.txt")
        
    with open(os.path.join(data_dir, "experimentData.txt")) as f:
        lines = f.read().splitlines()
        line = lines[-1].split(' ')
    #line.pop()

    size = int(float(line[0]))
    overlapRatio = float(line[1])
    inputSphereRadius = float(line[5])
    #prefactors = [float(line[6]), float(line[7]), float(line[8]), float(line[9])]
    numberOfBalls = int(float(line[10]))
    edgeLength = float(line[11])
    numberOfRounds = int(float(line[14]))
    allgather_time = int(float(line[15]))
    start = float(line[18])

    DB_FILE = f"{db_name}.db"
    db = sqlite3.connect(DB_FILE)
    db.execute("PRAGMA foreign_keys=1")
    cur = db.cursor()

    # Process each computer rank file from test_0.txt up to test_(size-1).txt
    for rank_idx in range(size):
        file_name = f"test_{rank_idx}.txt"
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping rank {rank_idx}.")
            continue

        with open(file_path, "r") as f:
            lines = f.read().splitlines()

        print(f"Processing rank {rank_idx} file with {len(lines)} entries...")

        for row in lines:
            if not row.strip():  # Skip empty lines
                continue
                
            values = row.split(' ')
            if values[-1] == '':
                values.pop()

            # --- Map values exactly to your index map ---
            it_no_val      = int(float(values[0]))      # index 0
            temp_val       = float(values[1])           # index 1
            prob_val       = round(float(values[2]),3)  # index 2
            deltaE_val     = round(float(values[3]), 5) # index 3
            
            v_val          = float(values[4])        # index 4
            a_val          = float(values[5])        # index 5
            c_val          = float(values[6])        # index 6
            x_val          = float(values[7])        # index 7
            v0_val         = float(values[8])        # index 8
            a0_val         = float(values[9])        # index 9
            c0_val         = float(values[10])       # index 10
            x0_val         = float(values[11])       # index 11
            l_val          = float(values[12])       # index 12
            energy_val     = round(float(values[13]), 5)    # index 13 (E - E0)/L
            
            frame_num_val  = int(float(values[14]))  # index 14
            acc_ratio_val  = int(float(values[15]))  # index 15
            time_val       = float(values[16]) - start # index 16
            # rank_val     = int(values[17])         # index 17
            bin_idx_val    = float(values[18])       # index 18

            poly_file_name = f"test_{rank_idx}_{frame_num_val}.poly"
            poly_file_path = os.path.join(poly_dir, poly_file_name)
            coordinates_str = read_point_coordinates_from_polyFile(poly_file_path)

            # --- 1. INSERT INTO _experiments_all_curves ---
            cur.execute('''
                INSERT INTO _experiments_all_curves (
                    experimentID, pointCoordinates, frameNumber, rankNumber, temp, energy, time_in_sequence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                coordinates_str,
                frame_num_val,
                rank_idx, 
                temp_val,
                energy_val,
                round(time_val,1)
            ))

            curve_id = cur.lastrowid

            # --- 2. INSERT INTO _measures (Linked via curveID) ---
            cur.execute('''
                INSERT INTO _measures (
                    curveID, inputSphereRadius, V0, V, A0, A, C0, C, X0, X, L, edgeLength, numberOfBalls
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                curve_id,
                inputSphereRadius,
                round(v0_val, 4),
                round(v_val, 4),
                round(a0_val, 4),
                round(a_val, 4),
                round(c0_val, 4),
                round(c_val, 4),
                round(x0_val, 4),
                round(x_val, 4),
                round(l_val, 4),
                edgeLength,
                numberOfBalls
            ))

            #---3. INSERT INTO _experiment_stats_intra_rounds (Linked via curveID) ---
            cur.execute('''
                INSERT INTO _experiment_stats_intra_rounds (
                    curveID, rank, it_no, T, prob, deltaE, accept, energy, bin_index, time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                curve_id,
                rank_idx,
                it_no_val,
                temp_val,
                prob_val,
                deltaE_val,
                acc_ratio_val,
                energy_val, 
                bin_idx_val,
                time_val
           ))    

        db.commit()
        print(f"Rank {rank_idx} successfully committed to database.")

    db.close()
    return None

def load_temperature_data(experiment_id, db_name):

    DB_FILE = f"{db_name}.db"
    db = sqlite3.connect(DB_FILE)
    db.execute("PRAGMA foreign_keys=1")
    cur = db.cursor()

    # Mac/Linux safe paths
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    data_dir = os.path.join(base_dir, "data")

    if not os.path.exists(os.path.join(data_dir, "temperatures.txt")):
        raise FileNotFoundError(f"Could not find temperatures.txt")

    with open(os.path.join(data_dir, "temperatures.txt")) as f:
        lines = f.read().splitlines()
    temperatures = lines[0]
    #temperatures = list(map(float, lines[-1].split(' ')))

    if not os.path.exists(os.path.join(data_dir, "temp_exchange_stats.txt")):
        raise FileNotFoundError(f"Could not find temp_exchange_stats.txt")

    with open(os.path.join(data_dir, "temp_exchange_stats.txt")) as f:
        lines = f.read().splitlines()

        for row in lines:
            if not row.strip():  # Skip empty lines
                continue
                
            values = row.split(' ')
            if values[-1] == '':
                values.pop()

            # --- Map values exactly to your index map ---
            round_nbr      = int(float(values[0]))
            #evenOrOdd     = int(float(values[1]))
            size           = int(float(values[2]))
            binIndices     = ' '.join(values[3:3+size:])
            acceptOrNot    = ' '.join(values[3+size:3 +2*size:])
            prob           = ' '.join([str(round(float(values[3+2*size+i]), 4)) for i in range(size)])
            energy         = ' '.join([str(round(float(values[3+3*size+i]), 4)) for i in range(size)])
    
            cur.execute('''
                INSERT INTO _experiment_stats(
                    experimentID, round_nbr, temperatures, prob, acceptOrNot, binIndices, energy
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id,
                round_nbr,
                temperatures,
                prob, 
                acceptOrNot,
                binIndices,
                energy
            ))

    db.commit()
    db.close()
    return None

def email_completion(db_name, experiment_id, pdf_file=''):
    msg = EmailMessage()
    msg["From"] = "rhoslyn.coles@mathematik.tu-chemnitz.de"
    msg["To"] = "rhoslyn.coles@mathematik.tu-chemnitz.de"
    msg["Subject"] = "job completion on"+socket.gethostname()

    m=f"Hello job with {db_name} just finished ;-) \n\n"

    if pdf_file=='':
        m+="problem with evaluation so maybe something else didn't work"
    else:
        m+="see attachment with pdf file \n"
 
    msg.set_content(m)
 
    if pdf_file!='':
        m+="see attachment with pdf file \n"
        with open(pdf_file, "rb") as f:
            msg.add_attachment(f.read(),maintype="application",subtype="pdf",filename=pdd_file)
    # send via local sendmail
    p = subprocess.Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=subprocess.PIPE)
    p.communicate(msg.as_bytes())

    return None

def do():
    """
    -----------------------------------------------------------------
    not sure how this method should be, on the one hand you want to add in details, like the step number, the temperature range, the date started the computer the experiment is being run on... so it seems like this one should be a centralised database...
    """
    #db_name, experiment_id = load_experiment_data()
    db_name, experiment_id = "circleTB", 1
    print(db_name, experiment_id)
   # load_curve_data(experiment_id, db_name)
   # output_name = 'results_'+str(experiment_id) #maybe structure_rs_eta...
   # pm.execute_notebook('evaluate_experiment.ipynb',output_name+'.ipynb',  parameters={"db_name" :db_name})
   # subprocess.run(["jupyter", "nbconvert", "--to", "pdf", output_name+".ipynb"])
   # load_temperature_data(experiment_id, db_name)
   # #TOFU --> delete data folder --> do something about initial configs?
    email_completion(db_name, experiment_id, pdf_file='')

    return None

do()
