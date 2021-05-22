# Test of basic application usage
import pytest
import os
from src.view.view import View
from src.presenter.presenter import Presenter
from PyQt5 import QtCore


@pytest.fixture
def app():
    # Launch app
    view = View()
    Presenter(view)
    return view


def test_app_launch(qtbot, app):
    qtbot.addWidget(app)
