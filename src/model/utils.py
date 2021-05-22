import sqlite3
import pandas as pd
import numpy as np


def to_datetime(arg, formats, force=False):
    res = []
    for v in arg:
        fit_format = False
        for format in formats:
            try:
                res.append(pd.to_datetime(v, format=format))
                fit_format = True
                break
            except ValueError:
                continue
        if not fit_format:
            if force:
                res.append(np.nan)
            else:
                raise ValueError("time data {} does not match any format".format(v))
    return res


class PySQL:
    def __init__(self, path):
        self.path = path
        self.bdd = None

    def connect(self):
        self.bdd = sqlite3.connect(self.path)

    def initialize(self, table_name, dataframe):
        dataframe.to_sql(table_name, con=self.bdd, if_exists="replace")

    def feed(self, table_name, dataframe):
        dataframe.to_sql(table_name, con=self.bdd, if_exists="append")

    def get_table_names(self):
        cmd = "SELECT * FROM sqlite_master WHERE type='table'"
        return self.execute(cmd)

    def get_colnames(self, table_name):
        cmd = "PRAGMA table_info({})".format(table_name)
        return self.execute(cmd)

    def execute(self, cmd):
        out = pd.read_sql_query(cmd, con=self.bdd)
        return out

    def close(self):
        self.bdd.close()
