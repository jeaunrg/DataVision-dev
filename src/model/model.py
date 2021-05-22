from src.model import utils
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import copy
import matplotlib
import matplotlib.pyplot as plt


class Model():

    def request_database(self, url, cmd=None):
        pysql = utils.PySQL(url)
        pysql.connect()
        if cmd is not None:
            outdf = pysql.execute(cmd)
        pysql.close()
        return outdf

    def describe_database(self, url):
        cmd = "SELECT * FROM sqlite_master WHERE type='table'"
        return self.request_database(url, cmd)

    def describe_table(self, url, table_name):
        cmd = "PRAGMA table_info({})".format(table_name)
        return self.request_database(url, cmd)

    def extract_from_database(self, url, table, columns):
        if table is None:
            raise ValueError("Empty table")
        cmd = "SELECT [{0}] FROM {1}".format("], [".join(columns), table)
        return self.request_database(url, cmd)

    def plot(self, df, X, Y, Z, xlabel, ylabel, zlabel, xlim, ylim, zlim, groupby):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df = df.set_index(groupby[0])
        df = df.sort_values(by=X)
        for i, ind in enumerate(np.unique(df.index)):
            x, y = df.loc[ind, X], df.loc[ind, Y]
            if not isinstance(x, pd.Series):
                x, y = [x], [y]

            ax.plot([i.total_seconds() for i in x], y)

        def timeTicks(x, pos):
            d = datetime.timedelta(seconds=x)
            return str(d)
        formatter = matplotlib.ticker.FuncFormatter(timeTicks)
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

        return ax

    def merge(self, dfs, *args, **kwargs):
        outdf = copy.copy(dfs[0])
        for df in dfs[1:]:
            outdf = outdf.merge(df, *args, **kwargs)
        return outdf

    def round(self, df, colname, mode='floor', decimal=1, freq='min'):
        """
        mode: {'floor', 'ceil', 'round'}, default='round'
        """

        if pd.api.types.is_float_dtype(df[colname]):
            if mode == 'round':
                df[colname] = eval("np.around(df[colname], {})".format(decimal))
            else:
                df[colname] = eval("np.{}(df[colname])".format(mode))
            return df
        elif pd.api.types.is_timedelta64_dtype(df[colname]):
            df[colname] = eval("df[colname].dt.{0}('{1}')".format(mode, freq))
        else:
            print(df[colname])
        return df

    def reform(self, df, colname_as_index, colname_as_header, colname_as_values):
        indexes = np.unique(df[colname_as_index])
        headers = np.unique(df[colname_as_header])
        df = df.set_index([colname_as_index, colname_as_header])
        new_df = pd.DataFrame(index=indexes, columns=headers)
        for i, h in df.index:
            values = df.loc[(i, h), colname_as_values]

            try:
                v = np.unique(values.values)
            except Exception:
                v = [values]
            if len(v) == 1:
                v = v[0]
            else:
                v = str(list(v))
            new_df.loc[i, h] = v
        # new_df = new_df.fillna("")
        new_df = new_df.reset_index()
        return new_df

    def load_data(self, path, separator="\t", decimal=",", header=None, encoding="latin-1", clean=True, sort=False):
        df = pd.read_csv(path, sep=separator, decimal=decimal, encoding=encoding, index_col=None, header=header)
        if clean:
            df = self.clean_dataframe(df)
        df = df.convert_dtypes()
        if sort:
            df = df.sort_index()
        return df

    def clean_dataframe(self, df):
        df = df.dropna(axis=0, how="all")
        df = df.dropna(axis=1, how="all")
        df = df.drop_duplicates()
        return df

    def rearrange(self, df, value_colname, new_colname=None, new_indname=None, out_as_dict=False):
        if new_indname is not None:
            df = df.set_index(new_indname)

        d = {}
        for ind in np.unique(df.index):
            d[ind] = {}
            colnames, values = df.loc[ind, new_colname], df.loc[ind, value_colname]
            if isinstance(colnames, (pd.Series, pd.DataFrame)):
                colnames, values = colnames.values, values.values
            else:
                colnames, values = [colnames], [values]
            for colname, value in zip(colnames, values):
                if isinstance(colname, np.ndarray):
                    colname = tuple(colname)
                if colname not in d[ind].keys():
                    d[ind][colname] = value
                elif isinstance(d[ind][colname], list):
                    d[ind][colname].append(value)
                else:
                    d[ind][colname] = [d[ind][colname], value]
        if not out_as_dict:
            d = pd.DataFrame.from_dict(d).T
            d = d.reset_index()
        return d

    def standardize(self, df, type_dict, format_dict={}, unit_dict={}, force={}):
        types = {'': None, 'integer': np.int64, 'float': np.float64, 'boolean': bool,
                 'string': str, 'datetime': datetime, 'timedelta': timedelta}
        for colname in type_dict:
            t = type_dict[colname]
            if types[t]:
                if t == 'datetime':
                    if ' OR ' in format_dict[colname]:
                        df[colname] = utils.to_datetime(df[colname], formats=format_dict[colname].split(' OR '),
                                                        force=force[colname])
                    else:
                        df[colname] = pd.to_datetime(df[colname], format=format_dict[colname],
                                                     errors='coerce' if force[colname] else 'raise')
                elif t == 'timedelta':
                    df[colname] = pd.to_timedelta(df[colname], unit=unit_dict[colname],
                                                  errors='coerce' if force[colname] else 'raise')
                else:
                    df[colname] = df[colname].astype(types[t])
        return df

    def apply_formula(self, df, formula, formula_name):
        formula = formula.replace("[", "df['")
        formula = formula.replace("]", "']")
        formula = formula.replace(" x ", " * ")
        try:
            df[formula_name] = eval(formula)
        except SyntaxError:
            raise SyntaxError("incorrect formula")
        return df

    def compute_stats(self, df, column, groupBy=None,
                      statistics=["count", "minimum", "maximum", "mean", "sum", "median", "std"],
                      ignore_nan=True, out_as_dict=False):
        """
        compute stats (recursively) if 'by' argument is set
        """
        if groupBy is None:
            if column is None:
                arr = df
            else:
                arr = df[column]
            if isinstance(arr, float):
                arr = [arr]
            else:
                arr = arr.values

            # compute statistics for and array
            d = {}
            for stat in statistics:
                if stat == "minimum":
                    d[stat] = np.min(arr)
                elif stat == "maximum":
                    d[stat] = np.max(arr)
                elif stat == "mean":
                    d[stat] = np.mean(arr)
                elif stat == "std":
                    d[stat] = np.std(arr)
                elif stat == "sum":
                    d[stat] = np.sum(arr)
                elif stat == "median":
                    d[stat] = np.median(arr)
                elif stat == "count":
                    d[stat] = len(arr)

            if not out_as_dict:
                d = pd.DataFrame.from_dict({"total": d}).T
            return d
        else:
            # compute stats recursively for each 'by'
            d = {}
            df = df.set_index(groupBy)
            for ind in np.unique(df.index):
                out = self.compute_stats(df.loc[[ind]], column, statistics=statistics,
                                         ignore_nan=ignore_nan, out_as_dict=True)
                if isinstance(out, type):
                    return out
                d[ind] = out
            if not out_as_dict:
                d = pd.DataFrame.from_dict(d).T
            d = d.reset_index()
            return d

    def get_selections(self, data, equal_to=None, different_from=None, higher_than=None, lower_than=None):
        selections = []
        if equal_to is not None:
            if not isinstance(equal_to, list):
                equal_to = [equal_to]
            preselection = []
            for et in equal_to:
                preselection.append(list((data == et).values))
            selections.append(list(np.logical_or.reduce(np.array(preselection))))
        if different_from is not None:
            if not isinstance(different_from, list):
                different_from = [different_from]
            preselection = []
            for df in different_from:
                preselection.append(list((data != df).values))
            selections.append(list(np.logical_and.reduce(np.array(preselection))))
        if higher_than is not None:
            selections.append(list((data >= higher_than).values))
        if lower_than is not None:
            selections.append(list((data <= lower_than).values))
        selections = np.array(selections)

        return selections

    def select_rows(self, df, column, equal_to=None, different_from=None,
                    higher_than=None, lower_than=None, logical="or"):
        selections = self.get_selections(df[column], equal_to, different_from, higher_than, lower_than)
        if logical == "and":
            selection = np.logical_and.reduce(selections)
        elif logical == "or":
            selection = np.logical_or.reduce(selections)
        if isinstance(selection, np.ndarray):
            df = df.loc[selection]
        return df

    def select_columns(self, df, columns):
        return df[columns]

    def save_data(self, dfs, path):
        if not isinstance(dfs, list):
            dfs = [dfs]
        for i, df in enumerate(dfs):
            if i > 0:
                df.to_csv(path+"_{}".format(i), sep='\t', decimal=",", encoding="latin-1")
            else:
                df.to_csv(path, sep='\t', decimal=",", encoding="latin-1")

    def fit_closest_event(self, events, events_datetime_colname, events_param_colname,
                          data, datetime_colname, value_colname,
                          on, delta=['before', 'after'], groupby=None, column_prefix=""):
        """
        This function find the values of parameters closest (in terms of time) to specified event

        Parameters
        ----------
        events: pd.DataFrame
            data of events
        events_datetime_colname: str
            name of the 'events' column which contains the events datetimes,
            the column must be of type datetime.datetime
        events_param_colname: str
            name of the 'events' column which contains the event name
        data: pd.DataFrame
            data of parameters
        datetime_colname: str
            name of the 'data' column which contains the parameters datetimes,
            the column must be of type datetime.datetime
        value_colname: str
            name of the 'data' column which contains the parameters values
        on: str
            common key column name for 'data' and 'events'
        delta: list, default=['before', 'after']
            if before inside this list, extract the closest parameter value before the event
            if after inside this list, extract the closest parameter value after the event
        groupby: str or None, default=None
            column name of 'events'. If not None apply this function for each unique value
            of the 'events' column
        column_prefix: str, default=""
            prefix for created column names

        Return
        ------
        outdf: pd.DataFrame
            data with events in lines and values/delays before and after each event

        Example
        -------


        """

        left_on, right_on = on, on
        if groupby is not None:
            # apply the 'fit_closest_event' function for each unique value of the
            # 'groupby' column'
            outdf = pd.DataFrame()
            data = data.set_index(groupby)

            # loop over 'gropby column unique values'
            params = np.unique(data.index)
            for param in params:
                groupby = None
                out = self.fit_closest_event(events, events_datetime_colname, events_param_colname, data.loc[param],
                                             datetime_colname, value_colname, on, delta, groupby, param)
                # concatenate single-parameter result with muli-parameters result
                outdf = pd.concat([outdf, out], axis=1)
        else:
            # initialize
            outdf = {}
            data_names = np.unique(data[right_on])
            events = events.set_index(left_on)
            events = events[[events_datetime_colname, events_param_colname]]
            data = data.set_index([right_on, datetime_colname])
            data = data[value_colname]

            # loop over key index (common to events and data)
            for name in np.unique(events.index):
                if name not in data_names:
                    continue
                subevents = events.loc[[name]]
                subdata = data.loc[name]

                # sort data in order to use ffill and backfill properly
                subdata = subdata[~subdata.index.duplicated(keep='first')].sort_index()

                # loop over events
                for i in range(len(subevents.index)):
                    eventsdate, eventsparam = subevents.iloc[i]
                    d = {}

                    # get value with datetime anterior and closest to eventsdate
                    if 'before' in delta:
                        try:
                            ind_before = subdata.index.get_loc(eventsdate, method='ffill')
                            d["_".join([column_prefix, "value before"])] = subdata.iloc[ind_before]
                            d["_".join([column_prefix, "delay before"])] = eventsdate - subdata.index[ind_before]
                        except KeyError:
                            pass

                    # get value with datetime posterior and closest to eventsdate
                    if 'after' in delta:
                        try:
                            ind_after = subdata.index.get_loc(eventsdate, method='backfill')
                            d["_".join([column_prefix, "value after"])] = subdata.iloc[ind_after]
                            d["_".join([column_prefix, "delay after"])] = subdata.index[ind_after] - eventsdate
                        except KeyError:
                            pass
                    outdf[(name, eventsparam, eventsdate)] = d

            # convert result to dataframe
            outdf = pd.DataFrame.from_dict(outdf, orient='index')
            # outdf.index.names = [left_on, events_param_colname, events_datetime_colname]
            # outdf = outdf.reset_index()

        return outdf
