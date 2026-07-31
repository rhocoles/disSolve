import sqlite3
import os
import time
from datetime import datetime
import socket
import papermill as pm
import subprocess
from email.message import EmailMessage
import math
import json
import argparse
import shutil
import zipfile


def format_timestamp_custom(unix_time):
    # Converts time.time() into 'day/month/year/hour/min/sec'
    return datetime.fromtimestamp(unix_time).strftime('%d/%m/%Y/%H/%M/%S')

def get_computer_name():
    # Grabs the hostname and ensures it is returned as a standard string
    computer_name = socket.gethostname()
    return str(computer_name)


def load_experiment_data(experimentID):

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
    db_name = str(line[16])#Structure
    alpha = str(line[17])
    start = format_timestamp_custom(float(line[18]))
    temperature_description = line[19]
    assert temperature_description in ("geometric", "linear", "temp_scan"), f"Unexpected temperature_description: {temperature_description!r}" #assert condition, message checks that condition is True; if it's not, it raises an AssertionError with message and stops execution
    initial_curve_configs = line[20]
    end = format_timestamp_custom(time.time())
    comp_name = get_computer_name()

    DB_FILE = f"{db_name}.db"
    db = sqlite3.connect(DB_FILE)
    db.execute("PRAGMA foreign_keys=1")
    cur = db.cursor()

    cur.execute('''
    UPDATE _experiments SET
        rs = ?,
        eta = ?,
        prefactors = ?,
        startDate = ?,
        completionDate = ?,
        size = ?,
        computerName = ?,
        T_bot = ?,
        T_top = ?,
        temperature_description = ?,
        number_of_rounds = ?,
        allgather_time = ?,
        initial_curve_configs = ?
        WHERE id = ?
        ''', (
        round(overlapRatio, 3),
        round(eta, 3),
        str(prefactors),
        start,
        end,
        size,
        comp_name,
        T_bot,
        T_top,
        temperature_description,
        numberOfRounds,
        allgather_time,
        initial_curve_configs,
        experimentID
    ))

    db.commit()
    db.close()

    return None

def read_poly_file(full_path_poly):
    with open(full_path_poly, 'r') as f:
        lines = f.read().splitlines()

    # split into POINTS and POLYS sections
    point_lines = lines[lines.index('POINTS') + 1: lines.index('POLYS')]
    strand_lines = lines[lines.index('POLYS') + 1: lines.index('END')]

    #points into a dict
    points = {}
    for line in point_lines:
        idx, rest = line.split(': ')
        coords = list(map(float,  rest.split(' c(')[0].split()))
        points[int(idx)] = coords

    #points into strand
    curve = []
    configType = []
    for line in strand_lines:
        indices = list(map(int, line.split(': ')[1].split()))
        if indices[0] == indices[-1]:
            configType.append('closed')
            indices = indices[:-1]
        else:
            configType.append('open')
        curve.append([points[i] for i in indices])

    numberOfBalls = sum(len(c) for c in curve)

    return curve, numberOfBalls, configType

def center_curve_and_compute_radius_of_gyration(curveData):
    all_points = [p for strand in curveData for p in strand]
    n = len(all_points)

    centroid = [
        sum(p[0] for p in all_points) / n,
        sum(p[1] for p in all_points) / n,
        sum(p[2] for p in all_points) / n,
    ]

    centered_curve = [
        [[p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]] for p in strand]
        for strand in curveData
    ]

    radius_gyration = math.sqrt(
        sum(p[0]**2 + p[1]**2 + p[2]**2 for strand in centered_curve for p in strand) / n
    )

    return centered_curve, radius_gyration

def compute_curve_length_and_edge_length(curveData, configType):
    total_length = 0.0
    total_edges = 0

    for strand, strand_type in zip(curveData, configType):
        n = len(strand)
        num_edges = n if strand_type == 'closed' else n - 1

        for i in range(num_edges):
            p1 = strand[i]
            p2 = strand[(i + 1) % n]  # wraps around back to start for closed strands
            edge_len = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
            total_length += edge_len

        total_edges += num_edges

    average_edge_length = total_length / total_edges if total_edges > 0 else 0.0

    return round(total_length, 3), round(average_edge_length, 5)


def load_curve_data(experimentID, db_name):
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
    eta = float(line[2])
    reachConstraint = float(line[3])
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
            temperature    = float(values[1])           # index 1
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
            #l_val          = float(values[12])       # index 12
            energy         = round(float(values[13]), 8)    # index 13 (E - E0)/L
            
            frameNumber  = int(float(values[14]))  # index 14
            acceptRatio  = int(float(values[15]))  # index 15
            time_val     = float(values[16]) - start
            rankNumber   = int(float(values[17]))
            bin_idx_val  = int(float(values[18]))      # index 18
            round_nbr    = int(float(values[19]))
            it_no_intra_round = int(float(values[20]))     

            minLocalRadiusCurvature = float(values[21])    
            selfDistance = float(values[22])

            poly_file_name = f"test_{rankNumber}_{frameNumber}.poly"
            poly_file_path = os.path.join(poly_dir, poly_file_name)
            curveData, numberOfBalls, configType  = read_poly_file(poly_file_path)
            curveData, Rg = center_curve_and_compute_radius_of_gyration(curveData)
            L, edgeLength = compute_curve_length_and_edge_length(curveData, configType)

            # --- 1. INSERT INTO _experiments_all_curves ---
            cur.execute('''
                INSERT INTO _experiments_all_curves (
                    experimentID, pointCoordinates, frameNumber, rankNumber, time_in_sequence, temperature, energy, L, edgeLength, numberOfBalls, reachConstraint, radiusGyration, minLocalRadiusCurvature, selfDistance 
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
            experimentID,
            json.dumps(curveData),
            frameNumber,
            rankNumber,
            time_val,
            temperature,
            energy,
            L,
            edgeLength,
            numberOfBalls,
            reachConstraint,
            round(Rg, 5),
            minLocalRadiusCurvature,
            selfDistance
            ))
            curve_id = cur.lastrowid

            # --- 2. INSERT INTO _measures (Linked via curveID) ---
            cur.execute('''
                INSERT INTO _measures (
                    curveID, experimentID, inputSphereRadius, V0, V, A0, A, C0, C, X0, X
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                curve_id,
                experimentID,
                inputSphereRadius,
                round(v0_val, 4),
                round(v_val, 4),
                round(a0_val, 4),
                round(a_val, 4),
                round(c0_val, 4),
                round(c_val, 4),
                round(x0_val, 4),
                round(x_val, 4),
            ))

            #---3. INSERT INTO _experiment_stats_intra_rounds (Linked via curveID) ---
            cur.execute('''
                INSERT INTO _experiment_stats_intra_rounds (
                    curveID, rank, it_no, T, prob, deltaE, accept, energy, bin_index, time, round_nbr, it_no_intra_round
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                curve_id,
                rankNumber,
                it_no_val,
                temperature,
                prob_val,
                deltaE_val,
                acceptRatio,
                energy, 
                bin_idx_val,
                time_val,
                round_nbr,
                it_no_intra_round
           ))    

        db.commit()
        print(f"Rank {rank_idx} successfully committed to database.")

    db.close()
    return None

def load_temperature_data(experimentID, db_name):

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
                experimentID,
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

def email_completion(db_name, experimentID, pdf_files=''):
    msg = EmailMessage()
    msg["From"] = "coles@math.tu-chemnitz.de"
    msg["To"] = "rhoslyn.coles@mathematik.tu-chemnitz.de"
    msg["Subject"] = "job completion on"+socket.gethostname()

    m=f"Hello experiment {experimentID} with {db_name} just finished on {get_computer_name()} ;-) \n\n"

    if pdf_files=='':
        m+="problem with evaluation so maybe something else didn't work"
    else:
        m+="see attachment with pdf file \n"
 
    msg.set_content(m)
 
    if pdf_files!='':
        m+="see attachment with pdf file \n"
        for pdf_file in pdf_files:
            with open(pdf_file, "rb") as f:
                msg.add_attachment(f.read(),maintype="application",subtype="pdf",filename=pdf_file)
    # send via local sendmail
    p = subprocess.Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=subprocess.PIPE)
    p.communicate(msg.as_bytes())

    return None

def do():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    DB_FILE = f"{args.name}.db"
    db = sqlite3.connect(DB_FILE)
    cur = db.cursor()
    cur.execute("SELECT id FROM _experiments ORDER BY startDate DESC LIMIT 1")
    experimentID = cur.fetchone()[0]
    db.close()

    load_experiment_data(experimentID)
    load_curve_data(experimentID, args.name)

    #move polyFiles with test_#rank_#frnbr.poly naming convention into polyFiles_{experimentID}
    archived_dir = f"polyFiles_{experimentID}"
    shutil.copytree("polyFiles", archived_dir)
    shutil.make_archive(archived_dir, 'zip', root_dir=".", base_dir=archived_dir)
    shutil.rmtree(archived_dir)

    #evaluate results
    output_name = 'results_'+str(experimentID) #maybe structure_rs_eta...
    pm.execute_notebook('evaluate_experiment.ipynb',output_name+'.ipynb',  parameters={"name" : args.name})
    subprocess.run(["jupyter", "nbconvert", "--to", "pdf", output_name+".ipynb", "--no-input"])
    results_attachment = [output_name+'.pdf']
    
    #temperature evaluation
    db = sqlite3.connect(DB_FILE)
    cur = db.cursor()
    cur.execute("SELECT temperature_description FROM _experiments WHERE id = ?", (experimentID,))
    temperature_description = cur.fetchone()[0] 
    db.close()
    output_name = 'temp_stats_'+str(experimentID)
    if temperature_description ==  "temp_scan":
        pm.execute_notebook('evaluate_experiment_temperature_scan.ipynb', output_name+'.ipynb',  parameters={"name" : args.name})
    else:
        load_temperature_data(experimentID, args.name)
        pm.execute_notebook('evaluate_experiment_temperatures.ipynb', output_name+'.ipynb',  parameters={"name" : args.name})
    subprocess.run(["jupyter", "nbconvert", "--to", "pdf", output_name+".ipynb", "--no-input"])

    #email results
    results_attachment.append(output_name+'.pdf')
    email_completion(args.name, experimentID, pdf_files=results_attachment)

    return None

do()
