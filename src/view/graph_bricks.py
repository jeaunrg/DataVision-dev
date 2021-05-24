from PyQt5 import QtWidgets, uic, QtCore, QtGui
import os
from src import RSC_DIR
from src.view import utils, ui
import pandas as pd
import numpy as np
import traceback


class QGraphicsNode(QtWidgets.QWidget):
    positionChanged = QtCore.pyqtSignal()
    focused = QtCore.pyqtSignal(bool)
    nameChanged = QtCore.pyqtSignal(str, str)
    saveDataClicked = QtCore.pyqtSignal()
    """
    movable widget inside the graph associated to a pipeline's step

    Parameters
    ----------
    graph: Graph
    type: str
        type of node associated to specific widget and functions
    name: str
        unique name
    parents: list of Node, default=[]
        nodes whose outputs are self input
    position: tuple, default=(0,0)
        position of the node in the graphic view

    """
    def __init__(self, graph, type, name, parents=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = graph
        self.type = type
        self.name = name
        self.parents = parents
        self.childs = []
        self.links = []
        self.color = None

        self.last_position = QtCore.QPointF(0, 0)
        self.initialPosition = None

        self._item = self.QCustomRectItem(self)
        self._proxy = QtWidgets.QGraphicsProxyWidget(self._item)
        self._proxy.setWidget(self)

        self.moveBy = self._item.moveBy
        self.pos = self._item.pos

        self.propagation_child = None

        self.setToolTip(type)
        self.initUI(name)

    def initUI(self, name):
        uic.loadUi(os.path.join(RSC_DIR, "ui", "Node.ui"), self)

        self.grap = utils.replaceWidget(self.grap, ui.QGrap())
        self.openButton.setText(name)
        self._item.setRect(QtCore.QRectF(self.geometry().adjusted(0, 0, 0, 0)))

        self.parameters = uic.loadUi(os.path.join(RSC_DIR, 'ui', 'modules', self.type+'.ui'))
        self.result = QtWidgets.QWidget()

        self.loading.setStyleSheet("QProgressBar { \
                                    background-color: transparent; \
                                    border-color: transparent; }")

        self.resize(250, 1)  # never resize to 0
        self.initConnections()
        self.setState()

    def resizeEvent(self, event):
        rect = self._item.rect()
        rect.setWidth(self.width())
        rect.setHeight(self.height())
        self._item.setRect(rect)
        return QtWidgets.QWidget.resizeEvent(self, event)

    def initConnections(self):
        self.positionChanged.connect(self.moveSelection)
        self.focused.connect(self.focusNode)

        self.selected.stateChanged.connect(self.changeChildSelection)

        self.openButton.mouseDoubleClickEvent = lambda e: self.graph.editNode(self)
        self.openButton.clicked.connect(self.openParametersButton.clicked.emit)
        self.openButton.clicked.connect(self.openResultButton.clicked.emit)

        self.openParametersButton.clicked.connect(self.openParameters)
        self.openResultButton.clicked.connect(self.openResult)

    class QCustomRectItem(QtWidgets.QGraphicsRectItem):
        """
        graphic item which allow to move the widget in graphic view

        Parameters
        ----------
        parent: QViewWidget
        """
        def __init__(self, parent):
            super().__init__()
            self.parent = parent
            self.setAcceptHoverEvents(True)
            self.setFlags(QtWidgets.QGraphicsItem.ItemIsMovable |
                          QtWidgets.QGraphicsItem.ItemIsFocusable |
                          QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges)

        def itemChange(self, change, value):
            if change == QtWidgets.QGraphicsItem.ItemPositionChange:
                self.parent.deltaPosition = value - self.pos()
                self.parent.positionChanged.emit()
            elif change == QtWidgets.QGraphicsItem.ItemVisibleChange:
                self.parent.positionChanged.emit()
            return QtWidgets.QGraphicsRectItem.itemChange(self, change, value)

    def enterEvent(self, event):
        self.focused.emit(True)
        return QtWidgets.QWidget.enterEvent(self, event)

    def leaveEvent(self, event):
        self.focused.emit(False)
        return QtWidgets.QWidget.leaveEvent(self, event)

    def setState(self, state=None):
        if state == 'loading' or state is None:
            col = QtGui.QColor(0, 0, 0, 0)
        elif state == 'valid':
            col = QtGui.QColor(0, 200, 0, 255)
        elif state == 'fail':
            col = QtGui.QColor(255, 0, 0, 255)
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Background, col)
        self.loading.setAutoFillBackground(True)
        self.loading.setPalette(pal)
        self.loading.setMaximum(int(state != 'loading'))

    def isSelected(self):
        return self.selected.isChecked()

    def addToScene(self, scene):
        scene.addItem(self._item)

    def get_parent_names(self):
        return [p.name for p in self.parents]

    def get_parent_name(self, i=0):
        return self.parents[i].name

    def moveSelection(self):
        if self is self.graph.focus:
            for node in self.graph.getSelectedNodes(exceptions=[self]):
                node.moveBy(self.deltaPosition.x(), self.deltaPosition.y())

    def setInitialPosition(self):
        self.initialPosition = self.pos()

    def changeChildSelection(self, state):
        if self.graph.holdShift:
            for child in self.childs:
                child.selected.setChecked(state)

    @property
    def mid_pos(self):
        return self.width()/2, self.height()/2

    def delete(self):
        """
        delete itself and all related graphic items (links and junctions)
        """
        # delete links
        while self.links:
            self.graph.scene.removeItem(self.links[0])
            self.links[0].delete()

        # delete wild widget if in dock
        if isinstance(self.parameters.parent(), QtWidgets.QDockWidget):
            self.parameters.parent().close()
        if isinstance(self.result.parent(), QtWidgets.QDockWidget):
            self.result.parent().close()

        self.graph.scene.removeItem(self._item)
        self._proxy.deleteLater()
        self.deleteLater()

    def focusNode(self, boolean):
        self.graph.setEnabledScroll(not boolean)
        self.graph.focus = self if boolean else None

    def getChilds(self):
        """
        get all children

        Return
        ------
        childs: list of Node

        """
        childs = self.childs
        for child in self.childs:
            childs += child.getChilds()
        return childs

    def rename(self, new_name):
        self.openButton.setText(new_name)
        self.nameChanged.emit(self.name, new_name)
        self.name = new_name

    def setColor(self, new_color):
        if isinstance(new_color, list):
            new_color = QtGui.QColor(*new_color)
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Background, new_color)
        for w in [self.header, self.footer]:
            w.setAutoFillBackground(True)
            w.setPalette(pal)
        self.color = new_color

    def addWidgetInDock(self, widget, side, unique=True):
        if widget is None:
            return
        dock = self.graph._view.addWidgetInDock(widget, side, unique)
        self.nameChanged.connect(lambda _, newname: dock.setWindowTitle(newname))
        dock.setWindowTitle(self.name)

    def setSettings(self, settings):
        if settings is None:
            return

        for name, w in self.parameters.__dict__.items():
            if name in settings['parameters']:
                utils.setValue(w, settings['parameters'][name])

        state = settings['state']
        self.graph.editNode(self, state['name'], state['color'])

    def getSettings(self):
        settings = {'parameters': {}}
        for name, w in self.parameters.__dict__.items():
            value = utils.getValue(w)
            if value is not None:
                settings['parameters'][name] = value
        settings['state'] = {'name': self.name,
                             'type': self.type,
                             'parents': [p.name for p in self.parents],
                             'position': [self.pos().x(), self.pos().y()],
                             'color': [self.color.red(), self.color.green(), self.color.blue(), self.color.alpha()]}
        return settings

    def openParameters(self):
        self.addWidgetInDock(self.parameters, QtCore.Qt.LeftDockWidgetArea)

    def openResult(self):
        self.addWidgetInDock(self.result, QtCore.Qt.RightDockWidgetArea)

    def updateResult(self, result):
        """
        This function create widget from result and show it. The created widget
        depends on the result type

        Parameters
        ----------
        result: any type data

        """
        # create the output widget depending on output type
        if result is None:
            return
        elif isinstance(result, Exception):
            new_widget = QtWidgets.QTextBrowser()
            new_widget.setPlainText("[{0}] {1}\n\n{2}".format(type(result).__name__, result,
                                    "".join(traceback.format_tb(result.__traceback__)[1:])))
            new_widget.setStyleSheet("color : red; ")
        else:
            if isinstance(result, (int, float, str, bool)):
                new_widget = self.computeTextWidget(result)
            elif isinstance(result, pd.DataFrame):
                new_widget = ui.QCustomTableWidget(result)
                new_widget.save.clicked.connect(self.saveDataClicked.emit)

        # replace current output widget with the new one
        self.result = utils.replaceWidget(self.result, new_widget)


def ceval(arg):
    try:
        return eval(arg)
    except (NameError, TypeError):
        return arg


class QGraphicsLink(QtWidgets.QGraphicsPolygonItem):
    """
    graphic arrow between two graphic points

    Parameters
    ----------
    parent/child: Node
        the two nodes to link
    width: float, default=5
        width of the arrow line
    arrowWidth: float, default=10
        width of the arrow head
    arrowLen: float, default=10
        length of the arrow head
    space: float, default=20
        space between arrow extremity and nodes
    color: QColor, default=QtGui.QColor(0, 150, 0)
        color of the arrow background
    borderWidth: float, default=2
        width of the arrow border
    borderColor: QColor, default=QtGui.QColor(0, 150, 0)
        color of the arrow border

    """
    def __init__(self, parent, child, width=5, arrowWidth=10, arrowLen=10, space=[0, 20],
                 color=QtGui.QColor(0, 150, 0), borderWidth=2, borderColor=QtGui.QColor(0, 150, 0)):
        super().__init__()
        self._parent = parent
        self._child = child
        self.setZValue(-1)
        self.setPen(QtGui.QPen(ceval(borderColor), borderWidth))
        self.setBrush(ceval(color))
        self.width = width
        self.arrowWidth = arrowWidth
        self.arrowLen = arrowLen
        self.space = space
        self.updatePos()

    def intersects(self, line, rect, ref_position):
        """
        This method find the intersection between widget rect and line
        by checking the intersection between line and each rect border line.
        As the line comes from inside the rect, only one intersection exists

        Parameters
        ----------
        line: QLineF
        rect: QRect
            rect of the widget
        ref_position: QPoint
            absolute position of the rect int the graph

        Return
        ------
        result: QPointF
            first position found of the intersection
        """
        points = [rect.bottomLeft(), rect.bottomRight(), rect.topRight(), rect.topLeft()]
        for i in range(4):
            border = QtCore.QLineF(ref_position + points[i-1], ref_position + points[i])
            try:
                intersection_type, intersection_point = line.intersects(border)
            except AttributeError:
                intersection_point = QtCore.QPointF()
                intersection_type = line.intersect(border, intersection_point)
            if intersection_type == QtCore.QLineF.BoundedIntersection:
                return intersection_point
        return QtCore.QPointF()

    def delete(self):
        """
        delete connection between link and parent/child
        """
        self._parent.positionChanged.disconnect(self.updatePos)
        self._child.positionChanged.disconnect(self.updatePos)
        self._parent.links.remove(self)
        self._child.links.remove(self)

    def updatePos(self):
        """
        This method create the arrow between child and parent
        """
        # build direction line
        r1, r2 = self._parent.rect(), self._child.rect()
        line = QtCore.QLineF(self._parent.pos() + r1.center(),
                             self._child.pos() + r2.center())

        # build unit vectors
        unit = (line.unitVector().p2() - line.unitVector().p1())
        normal = (line.normalVector().unitVector().p2() - line.normalVector().unitVector().p1())

        # get arrow point
        p1 = self.intersects(line, r1, self._parent.pos()) + unit * self.space[0]
        p2 = self.intersects(line, r2, self._child.pos()) - unit * self.space[1]
        p11 = p1 + normal * self.width
        p12 = p1 - normal * self.width
        p21 = p2 + normal * self.width - unit * self.arrowLen
        p22 = p2 - normal * self.width - unit * self.arrowLen
        p23 = p2 + normal * self.arrowWidth - unit * self.arrowLen
        p24 = p2 - normal * self.arrowWidth - unit * self.arrowLen

        # build arrow
        if np.sign((p22 - p12).x()) == np.sign(unit.x()) and np.sign((p22 - p12).y()) == np.sign(unit.y()):
            self.setPolygon(QtGui.QPolygonF([p11, p21, p23, p2, p24, p22, p12, p11]))
        else:
            self.setPolygon(QtGui.QPolygonF())
