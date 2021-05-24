import json
import os
from src import CONFIG_DIR, RSC_DIR
from src.utils import ceval, empty_to_none
from src.view.utils import replaceWidget
from src.presenter import utils
from PyQt5 import QtWidgets
from src.view import ui
import pandas as pd


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
        self.retro_propagation_stack = []
        self.init_view_connections()

    # ------------------------------ CONNECTIONS ------------------------------#
    def init_view_connections(self):
        self.modules = json.load(open(os.path.join(CONFIG_DIR, "modules.json"), "rb"))
        self._view.initModulesParameters(self.modules)
        self._view.graph.nodeAdded.connect(lambda m: self.init_module_connections(m))

    def init_module_connections(self, module):
        """
        initialize module parameters if necessary
        create connections between view widgets and functions

        Parameters
        ----------
        module_name: str
            name of the loaded module

        """
        parameters = self.modules[module.type]
        module._runners = []

        try:
            activation_function = eval('self.'+parameters['function'])
            module.parameters.apply.clicked.connect(lambda: activation_function(module))
        except AttributeError as e:
            print(e)

        module.saveDataClicked.connect(lambda: self.call_save_data(module))
        self.init_modules_custom_connections(module)

    def init_modules_custom_connections(self, module):
        nparents = len(module.parents)

        if module.type == "save":
            module.parameters.browse.clicked.connect(lambda: self.browse_savepath(module))

        elif nparents == 0:
            if module.type == "loadCSV":
                module.parameters.browse.clicked.connect(lambda: self.browse_data(module))

            elif module.type == "SQLrequest":
                module.parameters.browse.clicked.connect(lambda: self.browse_data(module))
                module.parameters.path.editingFinished.connect(lambda: self.update_sql_module(module))
                module.parameters.browse.clicked.connect(lambda s: self.update_sql_module(module))
                module.parameters.tableNames = replaceWidget(module.parameters.tableNames, ui.QGridButtonGroup())
                module.parameters.colnames = replaceWidget(module.parameters.colnames, ui.QGridButtonGroup())
                module.parameters.groupbox.hide()

        elif nparents == 1:
            colnames = []
            df = utils.get_data(module.get_parent_name())
            if isinstance(df, pd.DataFrame):
                colnames += list(df.columns)

            if module.type == "describe":
                module.parameters.column.clear()
                module.parameters.column.addItems([''] + colnames)
                module.parameters.group_by.clear()
                module.parameters.group_by.addItems([''] + colnames)

            elif module.type == "selectColumns":
                grid = ui.QGridButtonGroup(3)
                module.parameters.colnames = replaceWidget(module.parameters.colnames, grid)
                grid.addWidgets(QtWidgets.QPushButton, colnames)
                grid.checkAll()
                module.parameters.selectAll.clicked.connect(lambda s: grid.checkAll(True))
                module.parameters.deselectAll.clicked.connect(lambda s: grid.checkAll(False))

            elif module.type == "selectRows":
                module.parameters.column.clear()
                module.parameters.column.addItems([''] + colnames)

            elif module.type == "round":
                module.parameters.colname.clear()
                module.parameters.colname.addItems(colnames)

            elif module.type == "operation":
                def connectButton(but, addBrackets=False):
                    txt_format = "{0} [{1}]" if addBrackets else "{0} {1}"
                    txt = '/' if but.objectName() == 'divide' else but.text()
                    but.clicked.connect(lambda: module.parameters.formula.setText(
                                        txt_format.format(module.parameters.formula.text(), txt)))

                grid = ui.QGridButtonGroup(3)
                module.parameters.colnames = replaceWidget(module.parameters.colnames, grid)
                grid.addWidgets(QtWidgets.QPushButton, colnames, checkable=False)
                for button in grid.group.buttons():
                    connectButton(button, True)

                for button in [module.parameters.subtract, module.parameters.add,
                               module.parameters.multiply, module.parameters.divide,
                               module.parameters.AND, module.parameters.OR,
                               module.parameters.parenthesis_l, module.parameters.parenthesis_r]:
                    connectButton(button)

            elif module.type == "standardize":
                for i in range(module.parameters.form.rowCount()):
                    module.parameters.form.removeRow(0)
                for i, colname in enumerate(colnames):
                    line = ui.QFormatLine()
                    module.parameters.form.addRow(QtWidgets.QLabel(colname), line)
                    module.parameters.__dict__['format_line_{}'.format(i)] = line

        elif nparents == 2:
            list_colnames = []
            for name in module.get_parent_names():
                colnames = ['']
                df = utils.get_data(name)
                if isinstance(df, pd.DataFrame):
                    colnames += list(df.columns)
                list_colnames.append(colnames)

            if module.type == "merge":
                module.parameters.on.clear()
                module.parameters.on.addItems(list(set(list_colnames[0]) & set(list_colnames[1])))
                module.parameters.left_on.clear()
                module.parameters.left_on.addItems(list_colnames[0])
                module.parameters.right_on.clear()
                module.parameters.right_on.addItems(list_colnames[1])

            elif module.type == "timeEventFitting":
                module.parameters.event.clear()
                module.parameters.event.addItems(module.get_parent_names())
                module.parameters.params.clear()
                module.parameters.params.addItems(module.get_parent_names())
                module.parameters.params.setEnabled(False)

                def fillCombos(eventId):
                    paramsId = int(not eventId)
                    module.parameters.params.setCurrentIndex(paramsId)
                    module.parameters.groupBy.clear()
                    module.parameters.groupBy.addItems(list_colnames[paramsId])
                    for cb in [module.parameters.paramsKey, module.parameters.paramsDatetime,
                               module.parameters.paramsValue]:
                        cb.clear()
                        cb.addItems(list_colnames[paramsId])
                    for cb in [module.parameters.eventKey, module.parameters.eventDatetime,
                               module.parameters.eventName]:
                        cb.clear()
                        cb.addItems(list_colnames[eventId])

                module.parameters.event.currentIndexChanged.connect(fillCombos)
                module.parameters.event.currentIndexChanged.emit(0)

        module.setSettings(self._view.settings['graph'].get(module.name))

    def update_sql_module(self, module):
        database_description = self._model.describe_database(module.parameters.path.text())

        # create table grid
        grid = ui.QGridButtonGroup(3)
        module.parameters.tableNames = replaceWidget(module.parameters.tableNames, grid)
        grid.addWidgets(QtWidgets.QRadioButton, database_description['name'])

        # update table colnames
        def update_columns(button, state):
            if state:
                colnames_grid = ui.QGridButtonGroup(3)
                module.parameters.colnames = replaceWidget(module.parameters.colnames, colnames_grid)
                if button is not None:
                    table_description = self._model.describe_table(module.parameters.path.text(), button.text())
                    colnames_grid.addWidgets(QtWidgets.QPushButton, table_description['name'])
                    colnames_grid.checkAll()
                module.parameters.groupbox.show()
                module.parameters.selectAll.toggled.connect(module.parameters.colnames.checkAll)

        grid.group.buttonToggled.connect(update_columns)
        grid.checkFirst()

    # --------------------- PRIOR  AND POST FUNCTION CALL ---------------------#
    def prior_manager(self, module):
        """
        This method is called by the utils.manager before of the function call

        Parameters
        ----------
        module: QWidget
        """
        module.setState('loading')
        for parent in module.parents:
            if not isinstance(utils.get_data(parent.name), pd.DataFrame):
                parent.propagation_child = module
                activation_function = eval('self.'+self.modules[parent.type]['function'])
                activation_function(parent)
                return False

        return True

    def post_manager(self, module, output):
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
        for child in module.childs:
            self.init_modules_custom_connections(child)

        # stop loading if one process is still running (if click multiple time
        # on the same button)
        are_running = [r.isRunning() for r in module._runners]
        if any(are_running):
            module.setState('loading')

        # continue process on propagation child module
        if module.propagation_child is not None:
            module.propagation_child.parameters.apply.clicked.emit()
        module.propagation_child = None

    # ----------------------------- utils -------------------------------------#
    def browse_data(self, module):
        """
        open a browse window to select a csv file or sql database
        then update path in the corresponding QLineEdit
        """
        dialog = QtWidgets.QFileDialog()
        filename, valid = dialog.getOpenFileName(module.graph, "Select a file...", self._data_dir)
        if valid:
            self._data_dir = os.path.dirname(filename)
            module.parameters.path.setText(filename)
            module.parameters.path.setToolTip(filename)

    def browse_savepath(self, module):
        """
        open a browse window to define the nifti save path
        """
        name = module.get_parent_name()
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(module.graph, 'Save file',
                                                                    os.path.join(self._out_dir, name), filter=".csv")
        self._out_dir = os.path.dirname(filename)
        module.parameters.path.setText(filename+extension)
        module.parameters.path.setToolTip(filename+extension)

    # ----------------------------- MODEL CALL --------------------------------#
    @utils.manager(True)
    def call_test_database_connection(self, module):
        function = self._model.request_database
        args = {'url': module.parameters.path.text()}
        return function, args

    @utils.manager(True)
    def call_extract_from_database(self, module):
        function = self._model.extract_from_database
        args = {"url": module.parameters.path.text(),
                "table": module.parameters.tableNames.checkedButtonText(),
                "columns": module.parameters.colnames.checkedButtonsText()}
        return function, args

    @utils.manager(True)
    def call_load_data(self, module):
        separator = module.parameters.separator.currentText()
        if separator == "{tabulation}":
            separator = '\t'
        elif separator == '{espace}':
            separator = ' '

        function = self._model.load_data
        args = {"path": module.parameters.path.text(),
                "separator": separator,
                "decimal": module.parameters.decimal.currentText(),
                "header": ceval(module.parameters.header.text()),
                "encoding": module.parameters.encoding.currentText(),
                "clean": module.parameters.clean.isChecked(),
                "sort": module.parameters.sort.isChecked()}
        return function, args

    @utils.manager(True)
    def call_standardize(self, module):
        type_dict, format_dict, unit_dict, force_dict = {}, {}, {}, {}
        for i in range(module.parameters.form.rowCount()):
            label = module.parameters.form.itemAt(i, 0).widget().text()
            hbox = module.parameters.form.itemAt(i, 1).widget().layout()
            type_dict[label] = hbox.itemAt(0).widget().currentText()
            force_dict[label] = hbox.itemAt(3).widget().isChecked()
            if type_dict[label] == 'datetime':
                format_dict[label] = hbox.itemAt(1).widget().text()
            if type_dict[label] == 'timedelta':
                unit_dict[label] = hbox.itemAt(2).widget().currentText()
            if type_dict[label] == '--':
                type_dict[label] = ''

        function = self._model.standardize
        args = {"df": utils.get_data(module.get_parent_name()),
                "type_dict": type_dict,
                "format_dict": format_dict,
                "unit_dict": unit_dict,
                "force": force_dict}
        return function, args

    @utils.manager(True)
    def call_describe(self, module):
        function = self._model.compute_stats
        args = {"df": utils.get_data(module.get_parent_name()),
                "column": empty_to_none(module.parameters.column.currentText()),
                "groupBy": empty_to_none(module.parameters.group_by.currentText()),
                "statistics": utils.get_checked(module.parameters, ["count", "minimum", "maximum",
                                                                    "mean", "sum", "median", "std"]),
                "ignore_nan": module.parameters.ignore_nan.isChecked()}
        return function, args

    @utils.manager(True)
    def call_select_rows(self, module):
        function = self._model.select_rows
        print(module.parameters.lower_than.text())
        print(ceval(module.parameters.lower_than.text()), type(ceval(module.parameters.lower_than.text())))
        args = {"df": utils.get_data(module.get_parent_name()),
                "column": module.parameters.column.currentText(),
                "equal_to":  ceval(module.parameters.equal_to.text()),
                "different_from":  ceval(module.parameters.different_from.text()),
                "higher_than": ceval(module.parameters.higher_than.text()),
                "lower_than": ceval(module.parameters.lower_than.text()),
                "logical": utils.get_checked(module.parameters, ["or", "and"])[0]}
        return function, args

    @utils.manager(True)
    def call_select_columns(self, module):
        function = self._model.select_columns
        args = {"df": utils.get_data(module.get_parent_name()),
                "columns": module.parameters.colnames.checkedButtonsText()}
        return function, args

    @utils.manager(True)
    def call_round(self, module):
        function = self._model.round
        args = {"df": utils.get_data(module.get_parent_name()),
                "colname": module.parameters.colname.currentText(),
                "mode": module.parameters.mode.currentText(),
                "decimal": module.parameters.decimal.value(),
                "freq": module.parameters.freq.currentText()}
        return function, args

    @utils.manager(True)
    def call_operation(self, module):
        function = self._model.apply_formula
        args = {"df": utils.get_data(module.get_parent_name()),
                "formula": module.parameters.formula.text(),
                "formula_name": module.name}
        return function, args

    @utils.manager(True)
    def call_save_data(self, module):
        function = self._model.save_data
        if module.type == 'save':
            args = {"path": module.parameters.path.text(),
                    "dfs": [utils.get_data(n) for n in module.get_parent_names()]}
        else:
            path = os.path.join(OUT_DIR, module.name + '.csv')
            args = {"path": path,
                    "dfs": [utils.get_data(module.name)]}
            module.result.path.setText(path)
            module.result.leftfoot.show()
        return function, args

    @utils.manager(True)
    def call_merge(self, module):
        function = self._model.merge
        args = {"dfs": [utils.get_data(n) for n in module.get_parent_names()],
                "how": module.parameters.how.currentText(),
                "on": ceval(module.parameters.on.currentText()),
                "left_on": ceval(module.parameters.left_on.currentText()),
                "right_on": ceval(module.parameters.right_on.currentText()),
                "left_index": module.parameters.left_index.isChecked(),
                "right_index": module.parameters.right_index.isChecked(),
                "sort": ceval(module.parameters.suffixes.text())}
        return function, args

    @utils.manager(True)
    def call_fit_time_events(self, module):
        function = self._model.fit_time_events
        args = {"events": utils.get_data(module.parameters.event.currentText()),
                "events_key": module.parameters.eventKey.currentText(),
                "events_datetime": module.parameters.eventDatetime.currentText(),
                "events_name": module.parameters.eventName.currentText(),
                "parameter": utils.get_data(module.parameters.params.currentText()),
                "parameter_key": module.parameters.paramsKey.currentText(),
                "parameter_datetime": module.parameters.paramsDatetime.currentText(),
                "parameter_values": module.parameters.paramsValue.currentText(),
                "parameter_groupby": module.parameters.groupBy.currentText(),
                "delta_before": ceval(module.parameters.deltaBefore.currentText()),
                "delta_after": ceval(module.parameters.deltaAfter.currentText())}
        return function, args
