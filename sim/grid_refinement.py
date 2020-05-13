#!/usr/bin/env python3
# coding: utf-8

import itertools
import numpy as np


def _refine_cell(max_depth, refinement_levels):
    """Refines a cell to `max_depth` depth by `refinement_levels` at each
    depth.

    This function effectively builds a tree, where the cell coordinates are
    keys and the values are subdicts, with the refined cell coordinates as
    keys. For example, the tree below presents the cell (45, 55) refined by
    one depth level using refinement levels (2, 2, 0).

    {(45, 55): {(0, 0): None,
                (0, 1): None,
                (1, 0): None,
                (1, 1): None},
     (46, 55): ...}

    The last refinement level is indicated by having `None` as values for the
    cell dictionaries

    Args
        max_depth (int): The maximum depth to refine the cell. Depth as in
        refinement tree depth.

        refinement_levels (tuple of int): I,J,K refinement levels to apply at
        each depth level of the refinement tree..

    Returns
        A tree with the refinement levels of the input cell.
    """
    if not max_depth:
        return None

    subcells = list(itertools.product(range(refinement_levels[0]),
                                      range(refinement_levels[1])))
    subcells = [c + (0,) for c in subcells]

    refinement_tree = {k: None for k in subcells}

    for k in refinement_tree.keys():
        refinement_tree[k] = _refine_cell(max_depth-1, refinement_levels)

    return refinement_tree


def _calc_refinement_level(coords, well_loc, radius_per_level, max_level):
    """Calculates the cell refinement level according to its location,
    proximity to the well, maximum refinement allowed and radius for
    each refinement level.

    Args
        coords (tuple of ints): Cell coordinates.
        well_loc (tuple of ints): Well coordinates.
        radius_per_level (int): The radius of each refinement depth.
        max_level (int): The maximum depth of refinement.

    Returns
        An integer representing the refinement level for the input cell.

    Raises:
        ValueError if the the coordinates dimensions don't match the well
        coordinates dimensions, e.g. `len(coords) != len(well_loc)`.
    """
    if len(coords) != len(well_loc):
        raise ValueError('Unmatching dimensions for cell or well coordinates.')

    diff = (np.abs(np.array(coords) - np.array(well_loc))) // radius_per_level
    return max_level - np.max(diff)


def refine_well(well, radius_per_level, nulls_map={}):
    """Given a single well position within a grid, refines the grid cells
    around the well location.

    The well locations must contain the refined coordinates, because they will
    be used to calculate the maximum refinement around the well.

    For now, we only refine cells in the I and J directions. The K direction
    remains untouched. Also, the refinement mask used is always (3,3,1) meaning
    that each cell wil lbe refined into 3 cells in the I direction and 3 cells
    in the J direction, thus, totalling 9 cells.

    Args
        wells (`well.Well`): The well object. We use the coordinates of the
        well to calculate the cells to be refined and the refinement level of
        the well to guide the refinement level of the cells themselves.

        radius_per_level (int): The radius of each refinement level.

        nulls_map (numpy.array or dict): The null cells map. This argument is
        used as a mask to avoid refining invalid cells. Cells containing a
        value 1 are considered valid, while cells with 0 are invalid. If a dict
        is given, and it is empty, then all cells are considered valid.

    Returns
        A dictionary with the base cells as keys and the children cells
        subtrees as values. See `_refine_cell` for documentation on the format
        of the tree.

    Raises:
        ValueError if the well is not located in a refined cell.
    """
    if not well.on_refined_cell:
        raise ValueError('Well is not located in a refined cell.')

    loc = well.base_loc
    max_depth = len(well.loc)

    j, i, k = loc

    jmin = j - max_depth * radius_per_level + 1
    jmax = j + max_depth * radius_per_level - 1
    imin = i - max_depth * radius_per_level + 1
    imax = i + max_depth * radius_per_level - 1

    to_refine = list(itertools.product(range(jmin, jmax + 1),
                                       range(imin, imax + 1)))

    if isinstance(nulls_map, dict) and not nulls_map:
        to_refine = set(to_refine)
    else:
        is_null = {n for n in to_refine if not nulls_map[n]}
        to_refine = set(to_refine) - is_null

    to_refine = [c + (0,) for c in to_refine]

    cell_to_level_map = {k: _calc_refinement_level(
        k, well.base_loc, radius_per_level, max_depth - 1) for k in to_refine}

    refinement_data = dict()
    for coords, depth in cell_to_level_map.items():
        if not depth:# or not nulls_map[coords[:-1]]:
            continue
        refinement_data[coords] = _refine_cell(depth, (3, 3, 1))

    return refinement_data


def _calc_tree_depth(refinement_tree):
    """Calculates the depth of a given refinement tree.

    Args
        refinement_tree (dict): The refinement tree. For more details about
        the tree, see: `refine_well`.

    Returns
        The depth of this tree.
    """
    if not refinement_tree:
        return 0

    depth = 0
    for k, v in refinement_tree.items():
        d = _calc_tree_depth(v)
        if d > depth:
            depth = d

    return 1 + depth


def refine_grid(wells, radius={}, nulls_map={}, default_radius=2):
    """Creates the refinement around a list of wells. For more details see:
    `refine_well`

    The function resolves conflicting refinements by maintaining the most
    refined tree at the conflicting cells.

    Args
        wells (list of `well.Well`): A list with all wells to use in the
        refinement process.

    KwArgs
        radius (dict): A dict with the well names and the radius of cells to
        refine per level of refinement.

        nulls_map (numpy.array or dict): A map with the valid cells. If an
        empty dict is given, then all cells are assumed as valid.

        default_radius (int): The default radius to use for wells not in
        `radius`.

    Returns
        An unified refinement tree for all wells given.
    """
    trees = {}
    for w in wells:
        try:
            r = radius[w.name]
        except KeyError:
            r = default_radius

        t = refine_well(w, r, nulls_map)

        to_check = {cell for cell in t if cell in trees}

        for cell in to_check:
            if _calc_tree_depth(t[cell]) > _calc_tree_depth(trees[cell]):
                trees[cell] = t.pop(cell)
            elif _calc_tree_depth(t[cell]) < _calc_tree_depth(trees[cell]):
                t.pop(cell)

        trees.update(t)

    return trees


def refinement_to_imex(refinement_tree, parent_block_addr_str):
    """Converts the grid refinement tree to a list of IMEX readable strings.

    Args
        refinement_tree (dict of dict): A refinement tree as generated by
        `refine_grid`.

        parent_block_addr_str (str): Address of the parent block in IMEX
        format, e.g. 45, 50, 1 / 2, 2, 1 / ...

    Returns
        A list of strings to be included directly on an IMEX input file.
    """
    if refinement_tree is None or refinement_tree.values() is None:
        return []

    strs = []
    fmt = 'REFINE {} INTO 3 3 1'

    for k, v in refinement_tree.items():
        if v is None:
            continue
        block_addr = '{},{},{}'.format(k[1]+1, k[0]+1, k[2]+1)
        if parent_block_addr_str:
            block_addr = parent_block_addr_str + ' / ' + block_addr
        strs.append(fmt.format(block_addr))
        strs.extend(refinement_to_imex(v, block_addr))

    return strs
