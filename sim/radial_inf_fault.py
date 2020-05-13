#!/usr/bin/env python
# coding: utf-8


import os
import json
from configparser import ConfigParser
from glob import glob
from enum import Enum
from multiprocessing import Lock, Pool
from warnings import warn
import sys

import numpy as np
import pandas as pd

from sim.reservoir import ReservoirGridInfo
from sim.analytical_sols import investigation_radius_solution
from sim.exceptions import MissingBaseSimulationFileError, MissingFieldError
from sim.grid_refinement import refine_grid
from sim.imex import (calc_pressure_derivate, call_cmgimex, call_cmgresults,
                      gen_rwd_file, gen_simulation_card,
                      parse_simulation_card, read_cmgresults_output)


class IncrementDirection(Enum):
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'


def gen_base_nulls(shape):
    return '*NULL *CON 1'


def my_json_encoder(o):
    if isinstance(o, (np.int, np.int32, np.int64)):
        return int(o)
    elif isinstance(o, (np.float, np.float32, np.float64)):
        return float(o)
    return json.dumps(o)


config = ConfigParser()
config.read('config.ini')
CMG_IMEX_PATH = config['CMG']['IMEX_PATH']
CMG_RESULTS_PATH = config['CMG']['RESULTS_PATH']

try:
    params_file = sys.argv[1]
except IndexError:
    params_file = 'radial_inf_parameters.json'

try:
    with open(params_file, 'r') as fin:
        data = json.load(fin)
        runs_info = data['runs']
except FileNotFoundError:
    warn('[main] Parameters file not found. Cannot continue.')
    raise

NUM_THREADS = config['default'].getint('NUM_THREADS')
DELTAP_THRESHOLD = config['default'].getfloat('DELTAP_THRESHOLD')
BALANCE_DATASET = config['default'].getboolean('BALANCE_DATASET')


def run_simulation(run_info):
    try:
        output_path = run_info['output_path']
    except KeyError:
        output_path = 'imex_output'

    try:
        csv_path = run_info['csv_results_path']
    except KeyError:
        csv_path = 'csv'

    refinement_radius_per_level = run_info['refinement_radius_per_level']
    base_file = run_info['reference_simulation_file']
    reservoir_grid = ReservoirGridInfo.from_dict(run_info['grid'])
    permi = run_info['reservoir']['PERMI']
    grid_dims = (reservoir_grid.nI, reservoir_grid.nJ)
    np.random.seed(run_info['seed'])
    position_config = run_info['position_config']

    try:
        fault_list = run_info['faults']
    except KeyError:
        fault_list = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    # READING THE SIMULATION CARD AND VARIABLES
    try:
        with open(base_file, 'r') as fin:
            basecard_contents = fin.readlines()
    except FileNotFoundError:
        raise MissingBaseSimulationFileError()

    try:
        params = parse_simulation_card(
            basecard_contents, ['REFPRES', 'DK', 'POR', 'PERMI'], 'data')
        refpres = params['REFPRES']
        por = params['POR']
        viso = 1
        params['VISO'] = viso
    except ValueError:
        raise

    wells_info = run_info['wells']
    wells = []
    for well_info in wells_info:
        try:
            w = Well.from_dict(well_info)
        except MissingFieldError:
            warn('[main] Error while loading a well. Jumping to the next one')
            continue
        wells.append(w)

    if not wells:
        raise ValueError('[main] No valid wells. Impossible to continue.')

    # Calculating the wells positions
    max_radius = investigation_radius_solution(params['TIME'][-1], permi, por,
                                               viso)
    well_locs = {}
    location_count = 0
    for w in wells:
        try:
            well_config = position_config[w.name]
        except KeyError:
            continue

        if well_config['block_increment'] < 0:
            location_inc = np.arange(start=-well_config['block_limit'],
                                     stop=0,
                                     step=-well_config['block_increment']) + 1
            location_inc = location_inc[::-1]
        else:
            location_inc = np.arange(start=0,
                                     stop=well_config['block_limit'],
                                     step=well_config['block_increment'])

        direction = IncrementDirection(well_config['direction'])
        z = np.repeat(0, len(location_inc))
        base_loc = w.base_loc

        if direction == IncrementDirection.HORIZONTAL:
            block_to_meters = location_inc * reservoir_grid.dJ
            if BALANCE_DATASET:
                in_radius = block_to_meters < (2 * max_radius)
                unique, counts = np.unique(in_radius, return_counts=True)
                unique_counts = dict(zip(unique, counts))
                location_inc = location_inc[:unique_counts[True]]

            locs = (np.repeat(base_loc[0], len(location_inc)),
                    location_inc + base_loc[1],
                    z)
        else:
            block_to_meters = location_inc * reservoir_grid.dI
            if BALANCE_DATASET:
                in_radius = block_to_meters < (2 * max_radius)
                unique, counts = np.unique(in_radius, return_counts=True)
                unique_counts = dict(zip(unique, counts))
                location_inc = location_inc[:unique_counts[True]]

            locs = (location_inc + base_loc[0],
                    np.repeat(base_loc[1], len(location_inc)),
                    z)
        well_locs[w.name] = list(zip(*locs))
        location_count = len(well_locs[w.name])

    if not fault_list:
        c = 'HOMOGENEOUS'
    else:
        c = 'FRACTURE' if permi < fault_list[0]['property']['PERMI'] else 'FAULT'

    classes = dict(zip(range(1, location_count+2), [c] * location_count))

    base_nulls = gen_base_nulls(grid_dims)
    base_nulls_path = os.path.join(output_path, 'base.NULLS')
    with open(base_nulls_path, 'w') as fout:
        print(base_nulls, file=fout)

    for i in range(location_count):
        sim_id = i + 1
        for w in wells:
            if w.name in well_locs:
                w.base_loc = well_locs[w.name][i]

        simfile_base_names = []
        grid_params = {**{'NULL': 'base.NULLS'}, **run_info['reservoir']}
        refinement_tree = refine_grid(wells,
                                      default_radius=refinement_radius_per_level)

        fname = os.path.join(output_path, '{}.dat'.format(sim_id))
        gen_simulation_card(basecard_contents, fname, grid_params,
                            {w.name: w for w in wells},
                            {'refinement': refinement_tree},
                            fault_list=fault_list)

        simfile_base_names.append(str(sim_id))

        # RUN IMEX AND RESULTS FOR ALL SIMULATIONS
        run_imex = set([os.path.join(output_path, n + '.dat')
                        for n in simfile_base_names])
        while run_imex:
            imex_ran = set()
            for f in run_imex:
                if not call_cmgimex(CMG_IMEX_PATH, f):
                    continue
                else:
                    imex_ran.add(f)
            run_imex -= imex_ran

        for f in simfile_base_names:
            rwd_name = os.path.join(output_path, f + '.rwd')
            irf_name = os.path.join(output_path, f + '.irf')
            rwo_name = os.path.join(output_path, f + '.txt')
            gen_rwd_file(rwd_name, irf_name, rwo_name, [w.name for w in wells])

        run_results = set([os.path.join(output_path, n + '.rwd')
                           for n in simfile_base_names])
        while run_results:
            results_ran = set()
            for f in run_results:
                if not call_cmgresults(CMG_RESULTS_PATH, f):
                    continue
                else:
                    results_ran.add(f)
            run_results -= results_ran

    # Remove temporary files generated by IMEX
    if 'remove_tmp_files' in run_info and run_info['remove_tmp_files']:
        tmp_files = glob(os.path.join(output_path, '*.rw*'))
        tmp_files.extend(glob(os.path.join(output_path, '*.log')))
        tmp_files.extend(glob(os.path.join(output_path, '*.out')))
        tmp_files.extend(glob(os.path.join(output_path, '*.*rf')))
        tmp_files.extend(glob(os.path.join(output_path, '*.dat')))
        tmp_files.extend(glob(os.path.join(output_path, '*.NULLS')))
        for f in tmp_files:
            os.remove(f)

    # Reading the results exported by CMG results and storing as CSVs
    wells_bhp = {w.name: pd.DataFrame() for w in wells}
    wells_dbhp = {w.name: pd.DataFrame() for w in wells}

    rwo_files = glob(os.path.join(output_path, '*.txt'))
    for f in rwo_files:
        base_name = os.path.basename(f)[:-4]
        data = read_cmgresults_output(f)

        for w in wells:
            wells_bhp[w.name][base_name] = data[w.name]
            w_data = np.zeros((len(data), 2))
            w_data[:, 0] = data.index.values
            w_data[:, 1] = data[w.name].values
            w_dbhp = calc_pressure_derivate(w_data, refpres)
            w_dbhp = pd.Series(data=w_dbhp[:, 2], index=w_dbhp[:, 0])
            wells_dbhp[w.name][base_name] = w_dbhp

        os.remove(f)

    for k in wells_bhp.keys():
        wells_bhp[k].to_csv(os.path.join(csv_path, '{}_bhp.csv'.format(k)))
        wells_dbhp[k].to_csv(os.path.join(csv_path, '{}_dbhp.csv'.format(k)))

    classes_df = pd.DataFrame.from_dict(
        classes, orient='index', columns=['class'])
    classes_df.to_csv(os.path.join(csv_path, 'classes.csv'),
                      index_label='scenario_id')

    if fault_list:
        fault_perm = fault_list[0]['property']['PERMI']
    else:
        fault_perm = permi

    return wells_bhp, wells_dbhp, classes_df, permi, fault_perm


def merge_dataset(full, to_add):
    """Merges the input data frames into a single data frame by renaming the
    `to_add` df columns sequentially to the `full` column names. Assuming that
    the column names are numbers, as in simulation run results.
    """
    cols_int = to_add.columns.values.astype(np.int)
    cols_new = list(cols_int + len(full.columns))
    cols_map = dict(zip(list(to_add.columns.values), cols_new))
    mapped_df = to_add.rename(columns=cols_map)
    return pd.concat([full, mapped_df], axis=1)


def merge_classes(full, to_add):
    """Merges a class data frame into another. It does so by renaming the index
    of `to_add` sequentially to the `full` index values. We assume that the
    index is numerical, of course.
    """
    idx_full = full.index.values
    idx_add = to_add.index.values
    idx_map = dict(zip(list(idx_add), list(idx_add + len(idx_full))))
    to_add = to_add.rename(index=idx_map)
    return pd.concat([full, to_add])


def init_child(lock_):
    global lock
    lock = lock_


lock = Lock()


def main():
    merged_wells_bhp = None
    merged_wells_dbhp = None
    merged_classes = None

    with Pool(processes=NUM_THREADS, initializer=init_child, initargs=(lock,)) as pool:
        results = pool.map_async(run_simulation, runs_info).get()
        for r in results:
            wells_bhp, wells_dbhp, classes, perm, fault_perm = r
            with lock:
                if not merged_wells_bhp:
                    merged_wells_bhp = wells_bhp
                    merged_wells_dbhp = wells_dbhp
                    merged_classes = classes
                else:
                    for k in wells_bhp.keys():
                        merged_wells_bhp[k] = merge_dataset(merged_wells_bhp[k],
                                                            wells_bhp[k])
                        merged_wells_dbhp[k] = merge_dataset(merged_wells_dbhp[k],
                                                             wells_dbhp[k])

                    merged_classes = merge_classes(merged_classes, classes)

    os.makedirs('merged', exist_ok=True)
    for k in merged_wells_bhp.keys():
        merged_wells_bhp[k].to_csv(os.path.join('merged', f'{k}_bhp.csv'))
        merged_wells_dbhp[k].to_csv(os.path.join('merged', f'{k}_dbhp.csv'))

    merged_classes.to_csv(os.path.join('merged', 'classes.csv'),
                          index_label='scenario_id')


if __name__ == '__main__':
    main()
