import json
import os
from src import CONFIG_DIR, RSC_DIR
from src.utils import ceval, empty_to_none
from src.view.utils import replaceWidget, getValue
from src.presenter import utils
from PyQt5 import QtWidgets
from src.view import ui
import pandas as pd


def prm(module, submodule=None):
    if submodule is None:
        return module.parameters
    else:
        return module.parameters.widgets[submodule]

def parent_names(module, submodule=None):
    return [p.name for p in module.parents]

def parent_name(module, submodule=None):
    if submodule is None or submodule == module.submodules[0]:
        return module.parents[0].name
    else:
        return module.name

class Presenter():
    """
    This class is part of the MVP app design, it acts as a bridge between
    the Model and the View

    Parameters
    ----------
    model: model.Model
    view: view.View

    """
    def __init__(self, view, model=None):
        self._model = model
        self._view = view
        self.threading_enabled = True
        self._data_dir = os.path.join(RSC_DIR, "data")
        self._out_dir = os.path.join(RSC_DIR, "data", "out")
        self.init_view_connections()

    # ------------------------------ CONNECTIONS ------------------------------#
    def init_view_connections(self):
        self.modules = json.load(open(os.path.join(CONFIG_DIR, "modules.json"), "rb"))
        self._view.initModulesParameters(self.modules)
        self._view.graph.nodeAdded.connect(lambda m: self.init_module_connections(m))

    def init_module_connections(self, module, submodule=None):
        """
        initialize module parameters if necessary
        create connections between view widgets and functions

        Parameters
        ----------
        module_name: str
            name of the loaded module

        """
        if submodule is not None:
            parameters = self.modules[submodule]
        else:
            parameters = self.modules[module.type]
            module.saveDataClicked.connect(lambda: self.call_save_data(module))
            if parameters.get('submodules') is not None:
                for sm in parameters.get('submodules'):
                    self.init_module_connections(module, sm)
                return

        # connect plot button
        def connect_quickplot():
            module.result.quickPlot.clicked.connect(lambda: self.call_quick_plot(*args))
        module.dfresultUpdated.connect(connect_quickplot)

        args = (module, submodule)
        try:
            activation_function = eval('self.'+parameters['function'])
            prm(*args).apply.clicked.connect(lambda: activation_function(*args))
        except AttributeError as e:
            print(e)

        self.init_module_custom_connections(*args)


    def init_module_custom_connections(self, module, submodule=None):
        args = module, submodule
        nparents = module.getNparents(submodule)
        mtype = submodule if submodule is not None else module.type

        if mtype == "save":
            prm(*args).browse.clicked.connect(lambda: self.browse_savepath(*args))

        elif nparents == 0:
            if mtype == "loadCSV":
                prm(*args).browse.clicked.connect(lambda: self.browse_data(False, *args))
                prm(*args).appendBrowse.clicked.connect(lambda: self.browse_data(True, *args))
                def show_details():
                    if prm(*args).path.text().endswith('.csv'):
                        prm(*args).detailsCSV.show()
                        prm(*args).detailsXLS.hide()
                    else:
                        prm(*args).detailsCSV.hide()
                        prm(*args).detailsXLS.show()
                prm(*args).path.editingFinished.connect(show_details)
                prm(*args).path.editingFinished.emit()
            elif mtype == "SQLrequest":
                prm(*args).browse.clicked.connect(lambda: self.browse_data(False, *args))
                prm(*args).path.editingFinished.connect(lambda: self.update_sql_module(*args))
                prm(*args).browse.clicked.connect(lambda s: self.update_sql_module(*args))
                prm(*args).tableNames = replaceWidget(prm(*args).tableNames, ui.QGridButtonGroup())
                prm(*args).colnames = replaceWidget(prm(*args).colnames, ui.QGridButtonGroup())
                prm(*args).groupbox.hide()

        elif nparents == 1:
            df = utils.get_data(parent_name(*args))
            colnames = []
            if isinstance(df, pd.DataFrame):
                colnames = [str(c) for c in df.columns]

            if mtype == "describe":
                prm(*args).column.clear()
                prm(*args).column.addItems([''] + colnames)
                prm(*args).group_by.clear()
                prm(*args).group_by.addItems([''] + colnames)

            elif mtype == "selectColumns":
                grid = ui.QGridButtonGroup(3)
                prm(*args).colnames = replaceWidget(prm(*args).colnames, grid)
                grid.addWidgets(QtWidgets.QPushButton, colnames)
                grid.checkAll()
                prm(*args).selectAll.clicked.connect(lambda s: grid.checkAll(True))
                prm(*args).deselectAll.clicked.connect(lambda s: grid.checkAll(False))

            elif mtype == "selectRows":
                prm(*args).column.clear()
                prm(*args).column.addItems([''] + colnames)

            elif mtype == "round":
                prm(*args).colname.clear()
                prm(*args).colname.addItems(colnames)

            elif mtype == "operation":
                def connectButton(but, addBrackets=False):
                    txt_format = "{0} [{1}]" if addBrackets else "{0} {1}"
                    txt = '/' if but.objectName() == 'divide' else but.text()
                    but.clicked.connect(lambda: prm(*args).formula.setText(
                                        txt_format.format(prm(*args).formula.text(), txt)))

                grid = ui.QGridButtonGroup(3)
                prm(*args).colnames = replaceWidget(prm(*args).colnames, grid)
                grid.addWidgets(QtWidgets.QPushButton, colnames, checkable=False)
                for button in grid.group.buttons():
                    connectButton(button, True)

                for button in [prm(*args).subtract, prm(*args).add,
                               prm(*args).multiply, prm(*args).divide,
                               prm(*args).AND, prm(*args).OR,
                               prm(*args).parenthesis_l, prm(*args).parenthesis_r]:
                    connectButton(button)

            elif mtype == "standardize":
                form = ui.QTypeForm()
                prm(*args).form = replaceWidget(prm(*args).form, form)
                form.addRows(colnames)

        elif nparents == 2:
            list_colnames = []
            for name in parent_names(*args):
                colnames = ['']
                df = utils.get_data(name)
                if isinstance(df, pd.DataFrame):
                    colnames += list(df.columns)
                list_colnames.append(colnames)

            if mtype == "merge":
                prm(*args).on.clear()
                prm(*args).on.addItems(list(set(list_colnames[0]) & set(list_colnames[1])))
                prm(*args).left_on.clear()
                prm(*args).left_on.addItems(list_colnames[0])
                prm(*args).right_on.clear()
                prm(*args).right_on.addItems(list_colnames[1])

            elif mtype == "timeEventFitting":
                pnames = parent_names(*args)
                prm(*args).A_event.clear()
                prm(*args).A_event.addItems(pnames)
                prm(*args).params.clear()
                prm(*args).params.addItems(pnames)
                prm(*args).params.setEnabled(False)

                def fillCombos(event):
                    if event not in pnames:
                        return
                    eventId = pnames.index(event)
                    paramsId = int(not eventId)
                    params = pnames[paramsId]
                    prm(*args).params.setCurrentText(params)
                    prm(*args).groupBy.clear()
                    prm(*args).groupBy.addItems(list_colnames[paramsId])
                    for cb in [prm(*args).paramsKey, prm(*args).paramsDatetime,
                               prm(*args).paramsValue]:
                        cb.clear()
                        cb.addItems(list_colnames[paramsId])
                    for cb in [prm(*args).eventKey, prm(*args).eventDatetime,
                               prm(*args).eventName]:
                        cb.clear()
                        cb.addItems(list_colnames[eventId])

                prm(*args).A_event.currentTextChanged.connect(fillCombos)
                prm(*args).A_event.currentTextChanged.emit(pnames[0])

        module.setSettings(self._view.settings['graph'].get(module.name))

    def update_sql_module(self, *args):
        database_description = self._model.describe_database(prm(*args).path.text())

        # create table grid
        grid = ui.QGridButtonGroup(3)
        prm(*args).tableNames = replaceWidget(prm(*args).tableNames, grid)
        grid.addWidgets(QtWidgets.QRadioButton, database_description['name'])

        # update table colnames
        def update_columns(button, state):
            if state:
                colnames_grid = ui.QGridButtonGroup(3)
                prm(*args).colnames = replaceWidget(prm(*args).colnames, colnames_grid)
                if button is not None:
                    table_description = self._model.describe_table(prm(*args).path.text(), button.text())
                    colnames_grid.addWidgets(QtWidgets.QPushButton, table_description['name'])
                    colnames_grid.checkAll()
                prm(*args).groupbox.show()
                prm(*args).selectAll.toggled.connect(prm(*args).colnames.checkAll)

        grid.group.buttonToggled.connect(update_columns)
        grid.checkFirst()

    # --------------------- PRIOR  AND POST FUNCTION CALL ---------------------#
    def prior_manager(self, module, submodule):
        """
        This method is called by the utils.manager before the function call

        Parameters
        ----------
        module: QWidget
        """
        # start loading
        module.setState('loading')

        # store signal propagation inside module
        parent_submodule = module.parentSubmodule(submodule)
        if parent_submodule is not None:
            if utils.get_data(module.name) is None:
                parent = module.parameters.widgets[parent_submodule]
                parent.propagation_child = module.parameters.widgets[submodule]
                parent.apply.clicked.emit()
                return False
        else:
            # store signal propagation
            for parent in module.parents:
                if utils.get_data(parent.name) is None:
                    parent.propagation_child = module
                    parent.parameters.apply.clicked.emit()
                    return False

        return True


    def post_manager(self, module, submodule, output):
        """
        This method manage the output of a model function based on the output type
        it is called by the utils.manager at the end of the model process

        Parameters
        ----------
        module: QWidget
        output: exception, str, pd.DataFrame, np.array, ...
        """
        if output is not None:
            utils.store_data(module.name, output)
        if isinstance(output, Exception):
            module.setState('fail')
        else:
            module.setState('valid')

        module.updateResult(output)

        childSubmodule = module.childSubmodule(submodule)
        if childSubmodule:
            self.init_module_custom_connections(module, childSubmodule)
        for child in module.childs:
            self.init_module_custom_connections(child)

        # stop loading if one process is still running (if click multiple time
        # on the same button)
        are_running = [r.isRunning() for r in module._runners]
        if any(are_running):
            module.setState('loading')

        # retropropagate inside module
        if submodule is not None and isinstance(module.parameters, ui.QMultiWidget):
            propagation_child = module.parameters.widgets[submodule].propagation_child
            if propagation_child is not None:
                if not isinstance(output, Exception):
                    propagation_child.apply.clicked.emit()
                module.parameters.widgets[submodule].propagation_child = None
                return

        # retropropagate signal between modules
        if module.propagation_child is not None:
            if not isinstance(output, Exception):
                module.propagation_child.parameters.apply.clicked.emit()
            module.propagation_child = None

    # ----------------------------- utils -------------------------------------#
    def browse_data(self, append=False, *args):
        """
        open a browse window to select a csv file or sql database
        then update path in the corresponding QLineEdit
        """
        dialog = QtWidgets.QFileDialog()
        filename, valid = dialog.getOpenFileName(args[0].graph, "Select a file...", self._data_dir)
        if valid:
            self._data_dir = os.path.dirname(filename)
            if append:
                filename = " ; ".join([prm(*args).path.text(), filename])
            prm(*args).path.setText(filename)
            prm(*args).path.setToolTip(filename)

    def browse_savepath(self, *args):
        """
        open a browse window to define the nifti save path
        """
        name = parent_name(*args)
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(args[0].graph, 'Save file',
                                                                    os.path.join(self._out_dir, name), filter=".csv")
        self._out_dir = os.path.dirname(filename)
        prm(*args).path.setText(filename+extension)
        prm(*args).path.setToolTip(filename+extension)

    # ----------------------------- MODEL CALL --------------------------------#
    @utils.manager(False)
    def call_quick_plot(self, module, submodule):
        function = self._model.quick_plot
        func_args = {'df': utils.get_data(module.name),
                     'index': ceval(module.result.Vheader.currentText())}
        return function, func_args

    @utils.manager(True)
    def call_test_database_connection(self, *args):
        function = self._model.request_database
        func_args = {'url': prm(*args).path.text()}
        return function, func_args

    @utils.manager(True)
    def call_extract_from_database(self, *args):
        function = self._model.extract_from_database
        func_args = {"url": prm(*args).path.text(),
                     "table": prm(*args).tableNames.checkedButtonText(),
                     "columns": prm(*args).colnames.checkedButtonsText()}
        return function, func_args

    @utils.manager(True)
    def call_load_data(self, *args):
        separator = prm(*args).separator.currentText()
        if separator == "{tabulation}":
            separator = '\t'
        elif separator == '{espace}':
            separator = ' '
        path = prm(*args).path.text()
        if " ; " in path:
            path = path.split(' ; ')

        function = self._model.load_data
        func_args = {"path": path,
                     "separator": separator,
                     "decimal": prm(*args).decimal.currentText(),
                     "header": ceval(prm(*args).header.text()),
                     "encoding": prm(*args).encoding.currentText(),
                     "clean": prm(*args).clean.isChecked(),
                     "sort": prm(*args).sort.isChecked(),
                     "sheet": ceval(prm(*args).sheet.text())}
        return function, func_args

    @utils.manager(True)
    def call_standardize(self, *args):
        values = getValue(prm(*args).form)
        type_dict, format_dict, unit_dict, force_dict = {}, {}, {}, {}
        for k, (t, f, u, force) in values.items():
            if t != '--':
                type_dict[k], force_dict[k] = t, force
                if t == 'datetime':
                    format_dict[k] = f
                elif t == 'timedelta':
                    unit_dict[k] = u

        function = self._model.standardize
        func_args = {"df": utils.get_data(parent_name(*args)),
                     "type_dict": type_dict,
                     "format_dict": format_dict,
                     "unit_dict": unit_dict,
                     "force": force_dict}
        return function, func_args

    @utils.manager(True)
    def call_describe(self, *args):
        function = self._model.compute_stats
        func_args = {"df": utils.get_data(parent_name(*args)),
                     "column": empty_to_none(prm(*args).column.currentText()),
                     "groupBy": empty_to_none(prm(*args).group_by.currentText()),
                     "statistics": utils.get_checked(prm(*args), ["count", "minimum", "maximum",
                                                                  "mean", "sum", "median", "std"]),
                     "ignore_nan": prm(*args).ignore_nan.isChecked()}
        return function, func_args

    @utils.manager(True)
    def call_select_rows(self, *args):
        function = self._model.select_rows
        func_args = {"df": utils.get_data(parent_name(*args)),
                     "column": prm(*args).column.currentText(),
                     "equal_to":  ceval(prm(*args).equal_to.text()),
                     "different_from":  ceval(prm(*args).different_from.text()),
                     "higher_than": ceval(prm(*args).higher_than.text()),
                     "lower_than": ceval(prm(*args).lower_than.text()),
                     "logical": utils.get_checked(prm(*args), ["or", "and"])[0]}
        return function, func_args

    @utils.manager(True)
    def call_select_columns(self, *args):
        function = self._model.select_columns
        func_args = {"df": utils.get_data(parent_name(*args)),
                     "columns": prm(*args).colnames.checkedButtonsText()}
        return function, func_args

    @utils.manager(True)
    def call_round(self, *args):
        function = self._model.round
        func_args = {"df": utils.get_data(parent_name(*args)),
                     "colname": prm(*args).colname.currentText(),
                     "mode": prm(*args).mode.currentText(),
                     "decimal": prm(*args).decimal.value(),
                     "freq": prm(*args).freq.currentText()}
        return function, func_args

    @utils.manager(True)
    def call_operation(self, *args):
        function = self._model.apply_formula
        func_args = {"df": utils.get_data(parent_name(*args)),
                     "formula": prm(*args).formula.text(),
                     "formula_name": args[0].name}
        return function, func_args

    @utils.manager(True)
    def call_save_data(self, *args):
        function = self._model.save_data
        if args[1] == 'save' or args[0].type == 'save':
            args = {"path": prm(*args).path.text(),
                    "dfs": [utils.get_data(n) for n in parent_names(*args)]}
        else:
            path = os.path.join(self._out_dir, args[0].name + '.csv')
            func_args = {"path": path,
                         "dfs": [utils.get_data(args[0].name)]}
            args[0].result.path.setText(path)
            args[0].result.leftfoot.show()
        return function, func_args

    @utils.manager(True)
    def call_merge(self, *args):
        function = self._model.merge
        func_args = {"dfs": [utils.get_data(n) for n in parent_names(*args)],
                     "how": prm(*args).how.currentText(),
                     "on": ceval(prm(*args).on.currentText()),
                     "left_on": ceval(prm(*args).left_on.currentText()),
                     "right_on": ceval(prm(*args).right_on.currentText()),
                     "left_index": prm(*args).left_index.isChecked(),
                     "right_index": prm(*args).right_index.isChecked(),
                     "sort": ceval(prm(*args).suffixes.text())}
        return function, func_args

    @utils.manager(True)
    def call_fit_time_events(self, *args):
        round_freq, round_mode = None, None
        if prm(*args).round.isChecked():
            round_freq = prm(*args).roundFreq.currentText()
            round_mode = prm(*args).roundMode.currentText()

        function = self._model.fit_time_events
        func_args = {"events": utils.get_data(prm(*args).A_event.currentText()),
                     "events_key": prm(*args).eventKey.currentText(),
                     "events_datetime": prm(*args).eventDatetime.currentText(),
                     "events_name": prm(*args).eventName.currentText(),
                     "parameter": utils.get_data(prm(*args).params.currentText()),
                     "parameter_key": prm(*args).paramsKey.currentText(),
                     "parameter_datetime": prm(*args).paramsDatetime.currentText(),
                     "parameter_values": prm(*args).paramsValue.currentText(),
                     "parameter_groupby": prm(*args).groupBy.currentText(),
                     "delta_before": ceval(prm(*args).deltaBefore.currentText()),
                     "delta_after": ceval(prm(*args).deltaAfter.currentText()),
                     'round_freq': round_freq,
                     'round_mode': round_mode,
                     "drop_empty": prm(*args).dropEmpty.isChecked()}
        return function, func_args
