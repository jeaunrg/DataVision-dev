from PyQt5 import QtCore
from src import RESULT_STACK
import copy
import functools


class Runner(QtCore.QThread):
    """
    QThread that activate a function with arguments

    Parameters
    ----------
    target: function
    *args, **kwargs: function arguments
    """
    def __init__(self, target, *args, **kwargs):
        super().__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs

        # where the function result is stored
        self.out = None

    def run(self):
        self.out = self._target(*self._args, **self._kwargs)


def deleteRunner(module, runner):
    module._runners.remove(runner)
    del runner.out
    del runner


def manager(threadable=True):
    """
    this decorator manage threading

    Parameters
    ----------
    threadable: bool, default=True
        if True, the model function will be processed inside a QThread (if allowed)

    """
    def decorator(foo):
        @functools.wraps(foo)
        def inner(presenter, module, submodule=None):
            cont = presenter.prior_manager(module, submodule)
            if not cont:
                return
            function, args = foo(presenter, module, submodule)
            function = protector(function)

            # start the process inside a QThread
            if threadable and presenter.threading_enabled:
                runner = Runner(function, **args)
                if '_runners' not in module.__dict__:
                    module._runners = []
                module._runners.append(runner)
                runner.finished.connect(lambda: (presenter.post_manager(module, submodule, runner.out),
                                                 deleteRunner(module, runner)))
                runner.start()
            else:
                presenter.post_manager(module, submodule, function(**args))
        return inner
    return decorator


def protector(foo):
    def inner(*args, **kwargs):
        try:
            return foo(*args, **kwargs)
        except Exception as e:
            print(e)
            return e
    return inner


def get_unavailable(names):
    unvailable = []
    for name in names:
        if name not in RESULT_STACK:
            unvailable.append(name)
    return unvailable


def get_data(name):
    if name in RESULT_STACK:
        return copy.copy(RESULT_STACK[name])



def get_checked(widget, names=None):
    checked = []
    if names is None:
        for name, w in widget.__dict__.items():
            if w.isChecked():
                checked.append(name)
    else:
        for name in names:
            w = widget.__dict__[name]
            if w.isChecked():
                checked.append(name)
    return checked


def store_data(name, df):
    RESULT_STACK[name] = df
