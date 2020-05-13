#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import re
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from collections import OrderedDict
from copy import deepcopy

from warnings import warn

from sim.grid_refinement import refinement_to_imex


def read_simulation_card(path_to_card, param_tags):
    """
    Reads the contents of the simulation card and calls
    "parse_simulation_card".

    Parameters
    ----------
    path_to_card <str>: The path to the simulation card.
    param_tags <list of str>: List of parameters to look for.

    Returns
    -------
    A dictionary where the param_tags are the keys and the values are the ones
    found in the card. Returns an empty dict if the file is invalid.

    Raise
    -----
    ValueError if the path or the param_tags are empty.
    """
    if not path_to_card:
        raise ValueError('Invalid path to simulation card')
    if not param_tags:
        raise ValueError('No parameters to look for')

    try:
        with open(path_to_card, 'r') as fin:
            contents = fin.readlines()
    except FileNotFoundError:
        print('Simulation card no found at the specified path \"{}\"'.format(
            path_to_card))
        return {}

    return parse_simulation_card(contents, param_tags)


def parse_simulation_card(basecard_contents, param_tags, base_dir=None):
    """Parses the IMEX simulation card and extracts the desired parameters
    from it.
    Raises FileNotFoundError if the path is invalid.

    Parameters
    ----------
    :param basecard_contents: The contents of the simulation card.
    :type basecard_contents: list of str
    :param param_tags: The parameters to look for when scanning the
        simulation card.
    :type param_tags: list of str
    :parameter base_dir: The full path to the simulation card. This
    is prepended to any paths after an "INCLUDE" command. Optional
    :type base_dir: str or os.path

    Returns
    -------
    A dictionary with the data. The keys are parameter tags, while the values
    are the ones found in the input simulation card. Results of INCLUDE tags
    are returned always. In this case, the filename containing the data is
    returned as value and the property name is the key.
    e.g ['PERM'] = 'PERMI.txt'.

    :raise: FileNotFoundError if the path to any subfiles is invalid.
    :raise: ValueError if the basecard_contents is invalid (empty).
    """

    if not basecard_contents:
        raise ValueError('Invalid file contents')

    results = {'TIME': []}
    param_set = set(param_tags)
    jump_next = False
    for lineno, line in enumerate(basecard_contents):
        if jump_next:
            jump_next = False
            continue

        # Only parse non-comment lines.
        if '**' in line[0:2]:
            continue

        # Jump empty lines.
        lline = re.split(r'\s{1,}', line.strip())
        if not lline:
            continue

        if lline[0] in ['DI', 'DJ', 'DK'] and len(lline) <= 2:
            nline = basecard_contents[lineno + 1].strip().split('*')
            results[lline[0]] = [int(nline[0]), int(nline[1])]
            jump_next = True
            continue

        for param in param_set:
            try:
                val = float(lline[-1])
            except ValueError:
                val = lline[-1]
                if len(val) and val[0] == val[-1] and val.startswith("'"):
                    val = val[1:-1]

            if param == lline[0]:
                if param in results:
                    if not isinstance(results[param], list):
                        results[param] = [results[param]]
                    results[param].append(val)
                else:
                    results[param] = val

        if 'INCLUDE' in lline[0]:
            fname = lline[1][1:-1]
            if base_dir:
                fname = os.path.join(base_dir, fname)
            with open(fname, 'r') as fin:
                fline = fin.readline().strip()
            props = {'PERM', 'NULL'}
            for p in props:
                if p in fline:
                    results[p] = fname

        if 'TIME' in lline[0]:
            results['TIME'].append(float(lline[-1]))

    return results


def read_perm_file(path_to_perm_file, grid_dims):
    """
    Reads the permeability data from an input file and returns the numpy.matrix
    (or matrices) with the data.

    Parameters
    ----------
    path_to_perm_file: str
        The full path to the permeability file. Can only read data from an
        independent file, and not from a simulation card for now.
    grid_dims: tuple
        The grid's I, J, K dimensions in this order.

    Returns
    -------
    A single, or, at most, 3 numpy matrices with the permeability data.
    One for each axis. If one permeability axis is equal to another, then that
    axis will be replicated. If an axis is not returned, it means that it was
    not present in the file.
    """

    # In order to read the permeability file, we first scan it for the
    # PERM(I,J,K) tags and store their line numbers. Then, we check the
    # following tag, if it says "ALL", we get the strings from this line
    # number to the next one with a tag and use numpy.fromstring method
    # to convert it into a numpy array. We then reshape the resulting array
    # using the "grid_size" parameter and assign it to the return value.

    def get_perm_tag_locations(file_contents):
        """
        Scans the file contents and returns a dict with the line numbers where
        each tag occurs. Additionally, the dict also contains the location EOF
        under the "END" key. Also returns the order in which the tags
        occur.

        Parameters
        ----------
        file_contents: list of str
            The contents of the permeability file.

        Returns
        -------
        The dictionary with the tags as keys and the line numbers as values.
        A list with the order of the tags in the file.
        """
        perm_tags = {'PERMI', 'PERMJ', 'PERMK'}
        perm_locs = {'PERMI': -1, 'PERMJ': -1, 'PERMK': -1,
                     'END': len(file_contents)}
        perm_order = []
        for i, line in enumerate(file_contents):
            for perm in perm_tags:
                if perm in line:
                    perm_locs[perm] = i
                    perm_order.append(perm)
                    break
        perm_order.append('END')
        return perm_locs, perm_order

    def build_perm_data_dict(perm_data_dict):
        """
        Given the pre-built dict with the permeability results, this function
        scans the dict and fills the missing values.

        Parameters
        ----------
        perm_data_dict: dict
            The pre-built dict.

        Returns
        -------
        The fully built dict.
        """
        # First, we must build a dependency graph.
        parent = {}
        for k, v in perm_data_dict.items():
            if isinstance(v, str):
                parent[k] = 'PERM' + v[-1]

        # If all variables depend on another variable, then, we have no value
        # in any of them, so we return.
        if len(parent) == 3:
            raise ValueError(
                'ERROR: No permeability data to fill the variables. Aborting.')

        # Now, we try to shrink the tree. We can do a single-pass shrink, since
        # we have only three variables, and at most two that depend on other
        # values.
        for k, v in parent.items():
            if v in parent:
                parent[k] = parent[v]

        # Finally, we scan the original parameter dict and check if the current
        # key is in the parent dict we constructed. If so, then we retrieve the
        # value of its parent, else, we retrieve the value directly from the
        # input dict.
        perm_data = {}
        for k, v in perm_data_dict.items():
            key = k
            if k in parent.keys():
                key = parent[k]
            perm_data[k] = perm_data_dict[key]

        return perm_data

    try:
        contents = []
        with open(path_to_perm_file, 'r') as fin:
            contents = fin.readlines()

        perm_data = {}
        perm_locs, perm_order = get_perm_tag_locations(contents)

        for i, perm in enumerate(perm_order):
            if perm == 'END':
                break
            curr_loc = perm_locs[perm]
            line = contents[curr_loc]
            stat = re.split(r'\s', line.strip())[-1]
            # If the tag is "ALL", then there is permeability data for this
            # axis. Else, then this axis equals another axis' permeability.
            if stat == 'ALL':
                # Getting the location of the next tag.
                next_loc = perm_locs[perm_order[i + 1]]
                # Merging the lists into a single string (except for the
                # PERM* tags themselves, hence the +1 and -1 in the indices).
                perm_contents = ' '.join(contents[(curr_loc + 1):next_loc])
                # Using numpy to convert that string into a matrix.
                mat = np.fromstring(perm_contents, sep=' ')
                mat = mat.reshape(grid_dims)
                perm_data[perm] = mat
            else:
                # Here, we treat the case that the data to get may not be
                # present yet e.g. PERMI EQUALSJ but PERMJ is located
                # after PERMI. In this case, we only put the corresponding
                # EQUALS* tag as value in the results dictionary, and we will
                # process it before returning.
                perm_data[perm] = stat

        perm_data = build_perm_data_dict(perm_data)
        return perm_data
    except FileNotFoundError:
        print('ERROR: Permeability definition file not found at the specified path \"{}\"'.format(
            path_to_perm_file))
        return 0


def gen_simulation_card(basecard_contents, path_to_output_card, var_dict,
                        well_dict, grid_prop_dict=None, fault_list=None):
    """
    Generates a simulation card to be fed to CMG IMEX, given the output file
    path and the parameter dictionary.

    Parameters
    ----------
    basecard_contents: list[str]
        The contents of the simulation card to use as template.
    path_to_output_card: string
        The path to the output simulation card file.
    var_dict: dict(str, any)
        A dictionary with the variables to replace in the template card.
        The available keys and their respective values are:
        * PERM: str - Path to the file containing the permeability data;
        * NULL: str - Path to the file containing the valid cells data;
    well_dict: dict(str, :class:`Well`)
        A dictionary with the wells to be placed in the reservoir.
        The key/well pair is defined by the well name and Well instance.
    grid_prop_dict: dict(str, any), optional
        A dictionary with the grid properties of the reservoir. For now, the
        only allowed property is the grid refinement, keyed under 'refinement'.
    fault_list:
        A list with faults information. Each fault is a dict with one property
        value (usually PERMI) and the ranges. The faults are inserted by
        applying a MOD command to the desired property.
    """

    def get_var_locations(contents):
        """
        Scans the file contents and returns a dict with the line numbers where
        each tag occurs.

        Parameters
        ----------
        contents: list[str]
            The contents of the simulation card.

        Returns
        -------
        The ordered dictionary with the tags as keys and the line numbers as
        values.
        """
        var_locs = OrderedDict()
        for i, line in enumerate(contents):
            for var in var_dict.keys():
                if var in line:
                    var_locs[var] = i
                    break
            for name in well_dict.keys():
                sline = re.split(r'\s{1,}', line)
                if 'WELL' in sline[0] and name in sline[1]:
                    var_locs[name] = i
                    break
            if grid_prop_dict is not None and 'refinement' in grid_prop_dict:
                if 'REFINE' in line:
                    if 'refinement' in var_locs:
                        var_locs['refinement'].append(i)
                    else:
                        var_locs['refinement'] = [i]
                elif 'DTOP' in line:
                    if 'refinement' in var_locs:
                        var_locs['refinement'].append(i+2)
                    else:
                        var_locs['refinement'] = [i+2]
        return var_locs

    tag_locs = get_var_locations(basecard_contents)
    output_contents = deepcopy(basecard_contents)

    for k, v in tag_locs.items():
        if k in ['NULL', 'POR']:
            output_contents[v] = 'INCLUDE {}'.format(var_dict[k])
        if 'PERM'in k:
            if isinstance(var_dict[k], (float, int)):
                output_contents[v] = '{} CON {}'.format(k, var_dict[k])
            elif 'EQUALS' in var_dict[k]:
                output_contents[v] = '{} {}'.format(k, var_dict[k])
            else:
                output_contents[v] = 'INCLUDE {}'.format(var_dict[k])
        elif k in well_dict:
            for i in range(v + 1, len(output_contents)):
                if 'BHPDEPTH' in output_contents[i] or 'WELL' in output_contents[i]:
                    break
                output_contents[i] = ''
            output_contents[v] = str(well_dict[k])
        elif k in grid_prop_dict:
            for l in v:
                output_contents[l] = ''
            strs = grid_prop_dict['refinement']
            if isinstance(grid_prop_dict['refinement'], dict):
                strs = refinement_to_imex(grid_prop_dict['refinement'], '')
            output_contents[v[0]] = '\n'.join(strs)
        else:
            warn('[gen_simulation_card] Parameter "{}" not found.'.format(k))

    if fault_list:
        for f in fault_list:
            try:
                prop_val = list(f['property'].items())[0]
            except IndexError:
                warn('[gen_simulation_card] Invalid property for fault.')
                continue

            if len(f['coords']) != 3:
                warn('[gen_simulation_card] Not enough coordinates. Ensure that there are ranges for I, J and K dimensions.')
                continue

            irange, jrange, krange = f['coords']
            line = f'MOD {irange} {jrange} {krange} = {prop_val[1]}'
            output_contents.insert(tag_locs[prop_val[0]]+1, line)

            # Since we inserted a new line, must shift the ones after it by one
            for k, v in tag_locs.items():
                if isinstance(v, list):
                    continue
                if v > tag_locs[prop_val[0]]:
                    tag_locs[k] += 1

    with open(path_to_output_card, 'w+') as fout:
        for l in output_contents:
            print(l.strip(), file=fout)


def gen_perm_file(path_to_output, perm_dict):
    """
    Writes a permeability data file.

    Given the path to the output file, and a dictionary with the permeability
    data, this function writes a permeability file to be used with CMG IMEX.
    The dicitonary data keys are the PERM(I,J,K) tags, while the values are
    numpy matrices with the data. If two matrices are equal, then the contents
    of one will be written and the other will refer to the original
    (via EQUALS(I,J,K) tag).

    Parameters
    ----------
    path_to_output <str>: Path to the output simulation card.
    perm_dict <dict>: Permeability data dict.
    """

    perm_tags = {}
    if np.array_equal(perm_dict['PERMI'], perm_dict['PERMJ']) and np.array_equal(perm_dict['PERMI'], perm_dict['PERMK']):
        perm_tags['PERMI'] = 'ALL'
        perm_tags['PERMJ'] = 'EQUALSI'
        perm_tags['PERMK'] = 'EQUALSI'
    elif np.array_equal(perm_dict['PERMI'], perm_dict['PERMJ']):
        perm_tags['PERMI'] = 'ALL'
        perm_tags['PERMJ'] = 'EQUALSI'
        perm_tags['PERMK'] = 'ALL'
    elif np.array_equal(perm_dict['PERMI'], perm_dict['PERMK']):
        perm_tags['PERMI'] = 'ALL'
        perm_tags['PERMJ'] = 'ALL'
        perm_tags['PERMK'] = 'EQUALSI'
    elif np.array_equal(perm_dict['PERMK'], perm_dict['PERMJ']):
        perm_tags['PERMI'] = 'ALL'
        perm_tags['PERMJ'] = 'ALL'
        perm_tags['PERMK'] = 'EQUALSJ'
    else:
        perm_tags['PERMI'] = 'ALL'
        perm_tags['PERMJ'] = 'ALL'
        perm_tags['PERMK'] = 'ALL'

    with open(path_to_output, 'w+') as fout:
        for k, v in perm_tags.items():
            if v == 'ALL':
                print('{} {}'.format(k, v), file=fout)
                M = perm_dict[k]
                for k in range(M.shape[2]):
                    for i in range(M.shape[0]):
                        for j in range(M.shape[1]):
                            print(str(M[i, j, k]), flush=True, file=fout)
            else:
                print('{} {}'.format(k, v), file=fout)


def call_cmgimex(path_to_imex_exec, path_to_simulation_card):
    """
    Spawn a CMG IMEX process located at the input parameter path using the
    simulation card provided. Checks the resulting log file for errors.

    Parameters
    ----------
    path_to_imex_exec: str
        The full path to the IMEX executable e.g.
        'C:\\Program Files (x86)\\CMG\\EXEC\\bin\\msvc2010.exe'.
    path_to_simulation_card: str
        The path to the input simulation card.

    Returns
    -------
    True if the process executed successfuly, or False otherwise.
    """

    try:
        print('Calling {}'.format(path_to_imex_exec))
        print('\targ1 {}'.format(path_to_simulation_card))
        retcode = subprocess.call([path_to_imex_exec, '-f',
                                   path_to_simulation_card, '-dd',
                                   '-log', '-jacpar', '-parasol', '2', '-wait'],
                                  shell=True)
        if retcode < 0:
            print('ERROR [IMEX]. Process returned {}.'.format(-retcode))
            return False
    except OSError:
        print('ERROR [IMEX]. Execution failed.')
        return False

    log_file = path_to_simulation_card[:-3] + 'log'
    try:
        with open(log_file, 'r') as fin:
            log_contents = fin.readlines()
    except FileNotFoundError:
        print('Log file not found, could not check for simulation errors.')
        return False

    for l in log_contents:
        if 'FATAL ERROR' in l:
            return False

    return True


def gen_rwd_file(path, irf_in, rwo_out, wellnames):
    """
    Function to generate a CMG Results command file. For now, only the input
    and output files, as well as a single well may be specified.

    Parameters
    ----------
    path: str
        The path to write the resulting rwd file.
    irf_in: str
        The path of the input IRF file.
    rwo_out: str
        The path to the resulting CMG Results file.
    wellnames: str or list of str
        The name(s) of the well(s) to export data from.
    """
    with open(path, 'w+') as fout:
        wnames = ''
        for w in wellnames:
            wnames += "'{}' ".format(w)

        print('FILE   \'{}\''.format(irf_in), file=fout)
        print('OUTPUT   \'{}\''.format(rwo_out), file=fout)
        print(
            '\nPRECISION 12\nWIDTH 20\nLINES-PER-PAGE 1000000\nTABLE-WIDTH 10000', file=fout)
        print('\nUNITS MODSI\nTIME ON\nDATE OFF', file=fout)
        print('\nTABLE-FOR', file=fout)
        print(' WELLS {}'.format(wnames), file=fout)
        print(' COLUMN-FOR', file=fout)
        print('  PARAMETERS \'Well Bottom-hole Pressure\'', file=fout)
        print('TABLE-END', file=fout)


def call_cmgresults(path_to_results_exec,
                    path_to_rwd):
    """
    Spawn a CMG Results process located at the input parameter path, with the
    "rwd_in" model file.

    Parameters
    ----------
    path_to_results_exec: str
        The full path to the CMG Results executable e.g.
        'C:\\Program Files (x86)\\CMG\\EXEC\\bin\\Results.exe'.
    path_to_rwd: str
        Path to the input RWD file.

    Returns
    -------
    True if the process executed successfuly, False otherwise. Raises OSError
    if the process failed to execute (wrong path).
    """
    rwo_out = path_to_rwd[:-3] + 'rwo'
    try:
        print('Calling {}'.format(path_to_results_exec))
        print('\targ1 {}'.format(path_to_rwd))
        print('\targ2 {}'.format(rwo_out))
        retcode = subprocess.call([path_to_results_exec, '-f',
                                   path_to_rwd, '-o', rwo_out],
                                  shell=True)
        if retcode < 0:
            print('ERROR [RESULTS]. Process returned {}.'.format(-retcode))
            return False
    except OSError:
        print('ERROR [RESULTS]. Execution failed')
        return False

    return True


def read_cmgresults_output(path):
    """
    Reads the output file produced by the CMG Results program.

    Parameters
    ----------
    path: str
        The path to the output file.

    Returns
    A numpy.array containing the timestep/BHP data. If the file is invalid
    (does not exist), then a 1x1 array is returned.
    """
    filecontents = []
    try:
        with open(path, 'r') as fin:
            filecontents = [line.strip() for line in fin]
        if not filecontents:
            raise ValueError('File {} is empty.'.format(path))
    except FileNotFoundError:
        print('ERROR: Results file not found.')
        return np.zeros((1, 1))

    well_lims = OrderedDict()
    for l, txt in enumerate(filecontents):
        if 'WELL' in txt:
            well_name = re.split(r'\s{1,}', txt)[-1]
            well_lims[well_name] = l

    well_data = pd.DataFrame()
    for well_name, line_num in well_lims.items():
        prop_data = np.zeros((0, 2))
        for i in range(line_num + 4, len(filecontents)):
            line = filecontents[i].split()
            if 'TABLE' in line:
                well_data[well_name] = pd.Series(data=prop_data[:, 1],
                                                 index=prop_data[:, 0])
                break
            if not line:
                continue
            try:
                tsdata = list(map(float, line))
            except ValueError:
                continue

            prop_data = np.vstack((prop_data, np.array(tsdata)))
        # The last well may not be attributed to the dict in the loop above.
        well_data[well_name] = pd.Series(data=prop_data[:, 1],
                                         index=prop_data[:, 0])

    return well_data


def calc_pressure_derivate(bhp, ref_bhp):
    """
    Given the input pressure table, calculates the derivate using Bourdeut's
    algorithm.

    Parameters
    ----------
    bhp: numpy.array
        The Nx2 matrix containing the bottom-hole-pressure data (column 1)
        indexed by time (column 0).
    ref_bhp: number
        The reference bottom-hole pressure value to use when calculating the
        \delta{p} factor.

    Returns
    -------
    A numpy.array with the Nx3 values. The first columns contains the log(t)
    values, the 2nd column contains the \delta{p_wf} and the 3rd column
    contains the \delta{p_wf}' values.
    """

    deltap = np.zeros((bhp.shape[0] - 2, 3))
    T = bhp[:, 0]
    dP = ref_bhp - bhp[:, 1]

    for i in range(1, bhp.shape[0] - 1):
        d1 = math.log(T[i + 1] / T[i])
        c1 = math.log(T[i] / T[i - 1])
        c2 = math.log(T[i + 1] / T[i - 1])
        m1 = c1 / c2
        m2 = d1 / c2
        term1 = (dP[i + 1] - dP[i]) / d1 * m1
        term2 = (dP[i] - dP[i - 1]) / c2 * m2
        deltap[i - 1, 0] = math.log(T[i], 10)
        deltap[i - 1, 1] = dP[i]
        deltap[i - 1, 2] = term1 + term2

    return deltap


def eval_objective_function(ref_deltap, calc_deltap):
    """
    Evaluates the error between a reference deltap (*ref_deltap*) and a
    calculated deltap (*calc_deltap*).

    Parameters
    ----------
    ref_deltap: numpy.array
        Array containing the reference values
    calc_deltap: numpy.array
        Array containing the calculated values to be evaluated

    Returns
    -------
    fObj:
        Sum of squares of differences between the determine the error between
        the ref and the simulated data

    permAdj:
        Auxiliary value (%) to be add or decresed to the PERMK value
    """
    fObj = sum([math.pow((math.log10(i) - math.log10(j)) / (math.log10(i)), 2)
                for i, j in zip(ref_deltap, calc_deltap)])
    fAux = sum([(math.log10(i) - math.log10(j)) / (math.log10(i))
                for i, j in zip(ref_deltap, calc_deltap)])

    permAdj = 0
    if fObj > 2:
        permAdj = 0.1
    elif 1.0 < fObj <= 2.0:
        permAdj = 0.05
    elif 0.1 < fObj <= 1:
        permAdj = 0.02
    elif 0.05 < fObj <= 0.1:
        permAdj = 0.01
    else:
        permAdj = 0.005

    return fObj, fAux, permAdj


def plot_pressure_curves(fname, deltap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(range(deltap.shape[0]), deltap[:, 1], c='red', marker='o')
    ax1.set_title(r'P_wf \ log(t)')
    ax1.set_ylabel('P_wf')
    ax1.set_xlabel('timestep')
    ax1.autoscale()

    ax2.plot(range(deltap.shape[0]), deltap[:, 2], c='blue', marker='x')
    ax2.set_title(r"P_wf' \ log(t)")
    ax2.set_ylabel(r"P_wf'")
    ax2.set_xlabel('timestep')
    ax2.autoscale()

    fig.tight_layout()
    fig.savefig(fname, fmt='png', dpi=300)

    plt.close()


def optimize_permeability(simcard_contents, base_bhp, simulator_path,
                          **kwargs):
    """Given a template simulation card and the path to the measured
    bottom-hole pressure (BHP), this function generates a permeability map in
    the I, J and K directions, and optimizes it until the simulated BHP and the
    true BHP converge. The desired error can be given as an optional argument.

    :param simcard_contents: Contents of the input simulation card.
    :type simcard_contents: list[str]
    :param base_bhp: Array with the ground truth BHP values.
    :type base_bhp: numpy.array
    :param simulator_path: Full path to the simulator EXE file.
    :type simulator_path: str

    :kwargs: Optional keyword arguments.
    :param cb_notify_progress: A callback function that notifies the caller
    when an IMEX run is completed. This function receives two arguments: the
    run BHP and BHP derivate. Default value is None.
    :type cb_notify_progress: function
    :param base_dir: The base directory for this simulation card. Used when
    ran remotely. Default is the current directory.
    :type base_dir: str or os.path
    :param tol: The tolerance value for the simulation. Default value is 1.0.

    :return: The simulation timesteps, BHP values, delta(p_wf)' values and the
    objective function values for each simulation round. The first column of
    the BHP and delta(p_wf)' matrices contain the reference BHP and
    delta(p_wf)' data. The remaining columns contain the simulated data at
    each iteration.
    :rtype: list[float], numpy.matrix and numpy.matrix, list[float]

    :raise: ValueError if any of the required paths is empty, or the files are
            invalid.
    """

    if not simcard_contents:
        raise ValueError('Invalid simulation card contents.')
    if not simulator_path:
        raise ValueError('Invalid simulator executable path.')

    tol = kwargs['tol'] if 'tol' in kwargs else 1
    base_dir = kwargs['base_dir'] if 'base_dir' in kwargs else './'

    params = parse_simulation_card(simcard_contents, ['REFPRES', 'WELL'],
                                   base_dir=base_dir)
    if 'WELL' not in params.keys():
        raise ValueError('No wells found in the simulation card.')
    if 'REFPRES' not in params.keys():
        print('WARNING: Initial Well bottom-hole pressure value not found on simulation card. Using default value of \'300.0 md\'')
        params['REFPRES'] = 300

    # Loading the true pressure values and calculating \delta{w_p} and
    # \delta{w_p}'.
    ref_bhp = np.array((np.array(params['TIME']),
                        base_bhp))
    ref_bhp = ref_bhp.T
    ref_deltap = calc_pressure_derivate(ref_bhp, params['REFPRES'])

    filepaths = {
        'RWD': os.path.join(base_dir, 'bhp.rwd'),
        'BHP': os.path.join(base_dir, 'BHP_{}.txt'),
        'IRF': os.path.join(base_dir, 'loop_{}.irf'),
        'SIMCARD': os.path.join(base_dir, 'loop_{}.dat'),
    }

    gen_simulation_card(simcard_contents, filepaths['SIMCARD'].format(0),
                        {'Well-1': [], 'PERM': 'PERMI_0.txt'})
    gen_rwd_file(filepaths['RWD'], filepaths['IRF'].format(0),
                 filepaths['BHP'].format(0), params['WELL'])
    shutil.copy2(params['PERM'], os.path.join(base_dir, 'PERMI_0.txt'))

    BHP = np.matrix(ref_bhp[:, 1]).T
    dP = np.matrix(ref_deltap[:, 2]).T
    objective_function = []

    sign_changed = False
    old_faux = 0
    fobj = sys.float_info.max
    count_runs = 1

    while fobj > tol and not sign_changed:
        print('Run {}'.format(count_runs))
        # 3
        success = call_cmgimex(simulator_path,
                               filepaths['SIMCARD'].format(count_runs-1))
        if not success:
            raise ValueError('Did not run CMG IMEX successfuly. See log.')

        # 4
        success = call_cmgresults('report.exe', filepaths['RWD'])
        if not success:
            raise ValueError('Did not run CMG Results successfuly. See log.')

        # 5
        bhpdata = read_cmgresults_output(filepaths['BHP'].format(count_runs-1))
        BHP = np.hstack((BHP, np.matrix(bhpdata[:, 1]).T))

        # 6
        deltap = calc_pressure_derivate(bhpdata, params['REFPRES'])
        dP = np.hstack((dP, np.matrix(deltap[:, 2]).T))

        fobj, faux, adjaux = eval_objective_function(
            ref_deltap[:, 2], deltap[:, 2])
        print(fobj, faux, adjaux)
        objective_function.append(fobj)

        if old_faux != 0:
            if np.sign(old_faux) != np.sign(faux):
                sign_changed = True

        old_faux = faux

        permfile_split = params['PERM'].split('.')
        permfile_path = '{}_{}.txt'.format(permfile_split[0], count_runs-1)
        perm_data = read_perm_file(permfile_path, (30, 30, 1))

        if faux > 0:
            perm_data['PERMI'] += perm_data['PERMI'] * adjaux
            perm_data['PERMJ'] += perm_data['PERMJ'] * adjaux
            perm_data['PERMK'] += perm_data['PERMK'] * adjaux
        else:
            perm_data['PERMI'] -= perm_data['PERMI'] * adjaux
            perm_data['PERMJ'] -= perm_data['PERMJ'] * adjaux
            perm_data['PERMK'] -= perm_data['PERMK'] * adjaux

        permfile_path = '{}_{}.txt'.format(permfile_split[0], count_runs)
        gen_perm_file(permfile_path, perm_data)

        gen_simulation_card(simcard_contents, filepaths['SIMCARD'].format(count_runs),
                            {'Well-1': [], 'PERM': permfile_path})

        gen_rwd_file(filepaths['RWD'], filepaths['IRF'].format(count_runs),
                     filepaths['BHP'].format(count_runs), params['WELL'])

        count_runs += 1

        if 'cb_notify_progress' in kwargs:
            kwargs['cb_notify_progress'](bhpdata[:, 1], deltap[:, 2])

    return params['TIME'], BHP, dP, objective_function


def main():
    from dataloaders import fhfloader

    def placeholder(bhp, dbhp):
        print('You called?')

    try:
        with open(os.path.join('data', 'sample.dat'), 'r') as fin:
            simcard_contents = fin.readlines()
    except FileNotFoundError:
        raise

    try:
        # bhp = np.loadtxt(os.path.join('data', 'dp_verdade.txt'))
        bhp = fhfloader.load_fhf(os.path.join('data', 'dp_verdade.fhf'))
        bhp = bhp['data']['Well-1'][1][:, 1]
    except:
        raise

    time, bhp, dbhp = optimize_permeability(simcard_contents, bhp,
                                            cb_notify_progress=placeholder)
#    time = np.log10(np.array(time))
#    np.savetxt('logt.txt', time.T, delimiter=';')
#    np.savetxt('bhp.csv', bhp, delimiter=';')
#    np.savetxt('deltap.csv', dbhp, delimiter=';')

    # Simulation steps
    # 1) Read the input simulation card to get the well names, initial pressure values and oil flow rate;
    # 2) Read the correct pressure values from a file (data/dp_verdade.txt);
    # 3) Generate an initial PERM* map using kriging;
    # Do:
    #   4) Run IMEX on using the current parameters (PERM, QO);
    #   5) Generate the RWD file to export the BHP data of the wells;
    #   6) Run Results to export the BHP data for all wells;
    #   7) Read the BHP data for each well;
    #   8) Calculate the BHP derivates;
    #   9) Evaluate the error for each well;
    #  10) Adjust the permeabilities;
    # While error < tolerance.


if __name__ == '__main__':
    main()
