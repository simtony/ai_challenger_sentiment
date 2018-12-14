from __future__ import division
import datetime as datetime_
import logging
import codecs
import os
import csv
import numpy as np
import time
import inspect
import copy

from tabulate import tabulate

logger = logging.getLogger(__name__)


def current_datetime(date_sep='-', time_sep=':', date_time_sep='_'):
    """
    Helper function to show current date and time

    Returns:
        datetime_str: string shows current date and time
    """
    format_str = '%Y{0}%m{0}%d{1}%H{2}%M{2}%S'.format(date_sep, date_time_sep, time_sep)
    datetime_str = datetime_.datetime.strftime(datetime_.datetime.now(), format_str)
    return datetime_str


def line_count(filename, skip_empty=True):
    """
    Count lines of a file.

    Args:
        filename: filename
        skip_empty: whether to skip line with blank spaces

    Returns:
        num_lines

    """
    with open(filename, mode='r') as fin:
        if skip_empty:
            return sum(1 for line in fin if line.strip())
        else:
            return sum(1 for _ in fin)


def maybe_mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


class ColoredFormatter(logging.Formatter):
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    RESET_SEQ = "\033[0m"

    def __init__(self, to_file=False):
        self.to_file = to_file
        self.colors = {
            'WARNING': self.YELLOW,
            'INFO': self.CYAN,
            'DEBUG': self.WHITE,
            'CRITICAL': self.YELLOW,
            'ERROR': self.RED
        }
        msg = "[$BOLD%(levelname)s:%(filename)s:%(lineno)d:%(funcName)s$RESET]: %(message)s"
        if self.to_file:
            msg = msg.replace("$RESET", '').replace("$BOLD", '')
        else:
            msg = msg.replace("$RESET", self.RESET_SEQ).replace("$BOLD", self.BOLD_SEQ)
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        format_record = record
        levelname = record.levelname
        if not self.to_file and levelname in self.colors:
            levelname_color = self.COLOR_SEQ % (30 + self.colors[levelname]) + levelname
            format_record = copy.copy(record)
            format_record.levelname = levelname_color
        return logging.Formatter.format(self, format_record)


class TFlogFilter(logging.Filter):
    def filter(self, record):
        if "tensorflow" in record.pathname or "tf" in record.pathname:
            return False
        else:
            return True


class Timer(object):
    def __init__(self, format_str='%d:%02d:%02d'):
        """
        Args:
            format_str: format of hour-minute-second
        """
        self.format_str = format_str
        self._start = time.time()
        self._last = self._start

    def reset(self):
        """
        Reset timer.
        """
        self._start = time.time()
        self._last = self._start

    def tick(self):
        '''
        Get time elapsed from lass tick.

        Returns:
            a formatted time string
        '''
        elapse = time.time() - self._last
        self._last = time.time()
        return self._elapse_str(elapse)

    def tock(self):
        '''
        Get time elapsed from start or last reset.

        Returns:
            a formatted time string
        '''
        elapse = time.time() - self._start
        return self._elapse_str(elapse)

    def __enter__(self):
        invoke_info = inspect.stack()[1]
        invoke_str = "[{}:{}][{}]".format(os.path.basename(invoke_info[1]), invoke_info[2], invoke_info[3])
        logger.info("{}: start timing.".format(invoke_str))
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapse_str = self.tock()
        invoke_info = inspect.stack()[1]
        invoke_str = "[{}:{}][{}]".format(os.path.basename(invoke_info[1]), invoke_info[2], invoke_info[3])
        logger.info("{}: time elapsed: {}.".format(invoke_str, elapse_str))

    def _elapse_str(self, elapse):
        second = int(elapse % 60)
        minute = int((elapse // 60) % 60)
        hour = int(elapse // 3600)
        elapse_str = self.format_str % (hour, minute, second)
        return elapse_str


class CSVTracker(object):
    def __init__(self, fields, fmts, log_dir, filename, start_time, check_finite=False):
        self.fields = ['time'] + fields
        self.fmts = ['%s'] + fmts
        self.check_finite = check_finite
        path = os.path.join(log_dir, '%s.%s.csv' % (start_time, filename))
        fout = open(path, 'w')
        fout.write(codecs.BOM_UTF8)
        self.csv_writer = csv.writer(fout, dialect='excel', delimiter=',', lineterminator='\n')
        self.csv_writer.writerow(self.fields)
        self.timer = Timer()
        self._time_per_track = None

    @property
    def time_per_track(self):
        return self._time_per_track

    def track(self, fetch_dict):
        fetch_dict['time'] = self.timer.tock()
        self._time_per_track = self.timer.tick()
        logger.info(', '.join([field + ': ' + fmt % fetch_dict[field] for field, fmt in zip(self.fields, self.fmts)]))
        self.csv_writer.writerow([fetch_dict[field] for field in self.fields])
        if self.check_finite:
            for key, val in fetch_dict.items():
                if isinstance(val, np.float32) and not np.isfinite(val):
                    raise ValueError("Invalid value: %s: %s" % (key, str(val)))


class StatefulTracker(object):
    def __init__(self, cmp_field, fields, log_dir, start_time, filename):
        self.cmp_field = cmp_field
        self.fields = fields + ['time']
        path = os.path.join(log_dir, '%s.%s.csv' % (start_time, filename))
        fout = open(path, 'w')
        fout.write(codecs.BOM_UTF8)
        self.csv_writer = csv.writer(fout, dialect='excel', delimiter=',', lineterminator='\n')
        self.csv_writer.writerow(self.fields)
        self.timer = Timer()

        self.best = {}
        self.current = {}
        self.last = {}

        self._staled_tracks = 0
        self._improved = False
        self._time_per_track = None
        self._total_time = None

    @property
    def staled_tracks(self):
        return self._staled_tracks

    @property
    def time_per_track(self):
        return self._time_per_track

    def improved(self):
        return self._improved

    def track(self, fetch_dict):
        self._time_per_track = self.timer.tick()
        self._total_time = self.timer.tock()
        fetch_dict['time'] = self._time_per_track
        self.last = self.current
        self.current = fetch_dict
        if not self.best or self.best[self.cmp_field] < fetch_dict[self.cmp_field]:
            self.best = fetch_dict
            self._staled_tracks = 0
            self._improved = True
        else:
            self._staled_tracks += 1
            self._improved = False

        self.csv_writer.writerow([fetch_dict[field] for field in self.fields])
        entries = [['current'] + [self.current.get(field) for field in self.fields],
                   ['best'] + [self.best.get(field) for field in self.fields],
                   ['last'] + [self.last.get(field) for field in self.fields]]
        logger.info('\n' + tabulate(entries, ['name'] + self.fields, floatfmt=".4f", stralign="right"))
