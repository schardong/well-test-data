# coding: utf-8

from enum import Enum
from warnings import warn

from util.exceptions import MissingFieldError


class ReservoirKDir(Enum):
    UP = 'UP'
    DOWN = 'DOWN'


class ReservoirGridInfo(object):
    def __init__(self):
        self._num_cells = [1, 1, 1]
        self._cell_size = [1, 1, 1]
        self._kdir = ReservoirKDir.UP

    def __str__(self):
        di = "DI IVAR {}*{}\n".format(self.nI, self.dI)
        dj = "DJ JVAR {}*{}\n".format(self.nJ, self.dJ)
        dk = "DK KVAR {}*{}\n".format(self.nK, self.dK)
        if self.nK == 1:
            dk = "DK ALL {}*{}\n".format(self.nI * self.nJ, self.dK)
        dtop = "DTOP {}*0\n".format(self.nI * self.nJ)

        return "GRID VARI {} {} {}\n".format(self.nI, self.nJ, self.nK) + \
            "KDIR UP\n" + di + dj + dk + dtop

    @property
    def nI(self):
        return self._num_cells[0]

    @nI.setter
    def nI(self, nI):
        if nI <= 0:
            raise ValueError('[ReservoirGridInfo.nI] nI <= 0')
        self._num_cells[0] = nI

    @property
    def nJ(self):
        return self._num_cells[1]

    @nJ.setter
    def nJ(self, nJ):
        if nJ <= 0:
            raise ValueError('[ReservoirGridInfo.nJ] nJ <= 0')
        self._num_cells[1] = nJ

    @property
    def nK(self):
        return self._num_cells[2]

    @nK.setter
    def nK(self, nK):
        if nK <= 0:
            raise ValueError('[ReservoirGridInfo.nK] nK <= 0')
        self._num_cells[2] = nK

    @property
    def dI(self):
        return self._cell_size[0]

    @dI.setter
    def dI(self, dI):
        if dI <= 0:
            raise ValueError('[ReservoirGridInfo.dI] dI <= 0')
        self._cell_size[0] = dI

    @property
    def dJ(self):
        return self._cell_size[1]

    @dJ.setter
    def dJ(self, dJ):
        if dJ <= 0:
            raise ValueError('[ReservoirGridInfo.dJ] dJ <= 0')
        self._cell_size[1] = dJ

    @property
    def dK(self):
        return self._cell_size[2]

    @dK.setter
    def dK(self, dK):
        if dK <= 0:
            raise ValueError('[ReservoirGridInfo.dK] dK <= 0')
        self._cell_size[2] = dK

    @property
    def kdir(self):
        return self._kdir

    @kdir.setter
    def kdir(self, val):
        try:
            self._kdir = ReservoirKDir(val)
        except ValueError:
            warn('[ReservoirGridInfo.kdir] Invalid KDir, defaulting to "UP".')
            self._kdir = ReservoirKDir.UP

    def to_dict(self):
        return {
            'nI': self.nI,
            'nJ': self.nJ,
            'nK': self.nK,
            'dI': self.dI,
            'dJ': self.dJ,
            'dK': self.dK,
            'kdir': self.kdir.value
        }

    @staticmethod
    def from_dict(obj):
        fields = ['nI', 'nJ', 'nK', 'dI', 'dJ', 'dK', 'kdir']
        for f in fields:
            if f not in obj:
                err = '[ReservoirGridInfo.from_dict] Missing required field (%s)'.format(f)
                raise MissingFieldError(err)

        try:
            kdir = ReservoirKDir(obj['kdir'])
        except ValueError:
            warn('Invalid KDir, defaulting to "UP".')
            kdir = ReservoirKDir.UP

        r = ReservoirGridInfo()
        r.nI = obj['nI']
        r.nJ = obj['nJ']
        r.nK = obj['nK']
        r.dI = obj['dI']
        r.dJ = obj['dJ']
        r.dK = obj['dK']
        r.kdir = kdir
        return r
