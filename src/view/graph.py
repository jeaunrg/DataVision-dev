from PyQt5 import QtWidgets, QtCore, QtGui
from src.view import utils
from src.view.graph_bricks import QGraphicsNode, QGraphicsLink
from src import DEFAULT, RESULT_STACK
import copy


# stack under, raise, show

class QGraph(QtWidgets.QGraphicsView):
    """
    widget containing a view to display a tree-like architecture with nodes
    and branches

    Parameters
    ----------
    mainwin: MainWindow
        QMainWindow where the graph is displayed
    direction: {'horizontal', 'vertical'}, default='horizontal'
        direction of the pipeline;
        horizontal is left to right, vertical is top to bottom

    """
    nodeAdded = QtCore.pyqtSignal(QGraphicsNode)

    def __init__(self, mainwin, direction='vertical'):
        super().__init__()
        self._view = mainwin
        self.direction = direction
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.scene = QtWidgets.QGraphicsScene()
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setScene(self.scene)
        self.contextMenuEvent = lambda e: self.openMenu()
        self.setBackgroundBrush(eval(self._view.theme['background_brush']))

        self.selectAll = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+A'), self)
        self.selectAll.activated.connect(self.selectNodes)

        self.installEventFilter(self)
        self.holdShift = False
        self._mouse_position = QtCore.QPoint(0, 0)
        self.nodes = {}
        self.focus = None

    def bind(self, parent, child):
        """
        create a link between a parent and a child node

        Parameters
        ----------
        parent, child: Node
            nodes to visually bind
        """

        link = QGraphicsLink(parent, child, **self._view.theme['arrow'])

        parent.positionChanged.connect(link.updatePos)
        child.positionChanged.connect(link.updatePos)

        parent.links.append(link)
        child.links.append(link)
        self.scene.addItem(link)

    def setEnabledScroll(self, enable_scroll=True):
        """
        enable/disable view scrolling

        Parameters
        ----------
        enable_scroll: bool, default True
        """
        self.verticalScrollBar().setEnabled(enable_scroll)
        self.horizontalScrollBar().setEnabled(enable_scroll)

    def getSelectedNodes(self, exceptions=[]):
        """
        get all selected nodes (with ctrl+click shortcut)

        Return
        ------
        result: list of Node
        """
        return [n for n in self.nodes.values() if n.isSelected() and n not in exceptions]

    def eventFilter(self, obj, event):
        """
        manage keyboard shortcut and mouse events on graph view
        """
        if obj == self:
            if event.type() == QtCore.QEvent.Wheel:
                return True
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.unselectNodes()
            elif event.type() == QtCore.QEvent.KeyPress:
                if event.key() == QtCore.Qt.Key_Shift:
                    self.holdShift = True
                elif event.key() == QtCore.Qt.Key_Control:
                    self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            elif event.type() == QtCore.QEvent.KeyRelease:
                if event.key() == QtCore.Qt.Key_Shift:
                    self.holdShift = False
                elif event.key() == QtCore.Qt.Key_Control:
                    self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        return QtWidgets.QGraphicsView.eventFilter(self, obj, event)

    def unselectNodes(self):
        for node in self.nodes.values():
            node.selected.setChecked(False)

    def selectNodes(self, nodes=None):
        if nodes is None:
            nodes = self.nodes
        for node in nodes.values():
            node.selected.setChecked(True)

    def getUniqueName(self, name, exception=None):
        """
        find an unused name by adding _n at the end of the name

        Parameters
        ----------
        name: str
            default non-unique name of the node
        exception: None or str
            if new name is exception, keep it

        Return
        ------
        new_name: str
            unique name for the node

        """
        i = 1
        new_name = copy.copy(name)
        while new_name in self.nodes and new_name != exception:
            new_name = "{0}_{1}".format(name, i)
            i += 1
        return new_name

    def openMenu(self, node=None):
        """
        open menu on right-clic at clicked position

        Parameters
        ----------
        node: QGraphicsNode, default None
            if None open a menu with primary actions (load, ...)
            else open a menu with secondary actions (erosion, ...)

        """
        node = self.focus
        if node is None:
            acts = self._view.menu.get('primary')
            parents = []
        else:
            acts = self._view.menu.get('secondary')
            parents = [node] + self.getSelectedNodes()
            parents = list(set(parents))

        if acts is None:
            return

        def activate(action):
            type = action.text()
            if type in ["delete", "delete all"]:
                for p in parents:
                    self.deleteBranch(p)
            else:
                nparents = self._view.modules_parameters[action.text()]['nparents']
                if not parents:
                    self.addNode(action.text())
                else:
                    if parents and nparents in [1, None]:
                        for p in parents:
                            self.addNode(action.text(), [p])
                    elif nparents == -1 or nparents == len(parents):
                        self.addNode(action.text(), parents)
                    else:
                        self._view.setStatusTip("This node must have {0} \
                                                 parents, got {1}".format(nparents, len(parents)))

        menu = utils.menu_from_dict(acts, activation_function=activate)
        pos = QtGui.QCursor.pos()
        self._mouse_position = self.mapToScene(self.mapFromGlobal(pos))
        menu.exec_(QtGui.QCursor.pos())

    def editNode(self, node, new_name=None, new_color=None):
        self.renameNode(node, new_name)
        self.colorizeNode(node, new_color)

    def renameNode(self, node, new_name=None):
        if new_name is None:
            new_name, valid = QtWidgets.QInputDialog.getText(self, "user input", "new name",
                                                             QtWidgets.QLineEdit.Normal, node.type)
            if not valid:
                return
        new_name = self.getUniqueName(new_name, exception=node.name)
        self.nodes[new_name] = self.nodes.pop(node.name)
        if node.name in RESULT_STACK:
            RESULT_STACK[new_name] = RESULT_STACK.pop(node.name)
        node.rename(new_name)

    def colorizeNode(self, node, new_color=None):
        if new_color is None:
            new_color = QtWidgets.QColorDialog.getColor(node.color)
        if isinstance(new_color, list) or new_color.isValid():
            node.setColor(new_color)

    def deleteAll(self):
        while self.nodes:
            self.deleteBranch(list(self.nodes.values())[0])

    def deleteBranch(self, parent, childs_only=False):
        """
        delete node, its children and the associated data recursively

        Parameters
        ----------
        parent: QGraphicsNode
        child_only: bool, default=False
            if True do not delete the parent node else delete parent and children

        """
        if parent.name not in self.nodes:
            return
        # delete data
        if parent.name in RESULT_STACK:
            del RESULT_STACK[parent.name]
        # delete children if has no other parent
        for child in parent.childs:
            child.parents.remove(parent)
            if not child.parents:
                self.deleteBranch(child)
        # remove node from parent children
        for p in parent.parents:
            p.childs.remove(parent)
        # delete node and links
        parent.delete()
        del self.nodes[parent.name]

    def addNode(self, type, parents=None, position=None):
        """
        create a node with specified parent nodes

        Parameters
        ----------
        type: str
            type of node
        parents: list of QGraphicsNode or QGraphicsNode

        """
        if parents is None:
            parents = []
        if not isinstance(parents, list):
            parents = [parents]

        for i, parent in enumerate(parents):
            if isinstance(parent, str):
                parents[i] = self.nodes[parent]

        name = self.getUniqueName(type)
        node = QGraphicsNode(self, type, name, parents)
        node.addToScene(self.scene)

        if not parents:
            x, y = self._mouse_position.x(), self._mouse_position.y()
        else:
            max_x_parent = parents[0]
            max_y_parent = parents[0]
            for parent in parents:
                if parent.pos().x() > max_x_parent.pos().x():
                    max_x_parent = parent
                if parent.pos().y() > max_y_parent.pos().y():
                    max_y_parent = parent
                self.bind(parent, node)
                parent.childs.append(node)
            if self.direction == 'vertical':
                Xs = [c.pos().x() + c.width() for c in max_y_parent.childs if c is not node]
                x = max_y_parent.pos().x() if not Xs else max(Xs) + DEFAULT['space_between_nodes'][0]
                y = max_y_parent.pos().y() + max_y_parent.height() + DEFAULT['space_between_nodes'][1]
            else:
                Ys = [c.pos().y() + c.height() for c in max_x_parent.childs if c is not node]
                x = max_x_parent.pos().x() + max_x_parent.width() + DEFAULT['space_between_nodes'][0]
                y = max_x_parent.pos().y() if not Ys else max(Ys) + DEFAULT['space_between_nodes'][1]

        # set state
        if position is not None:
            x, y = position

        node.moveBy(x, y)
        self.nodes[name] = node
        self.nodeAdded.emit(node)

        node.setColor(self._view.modules_parameters[type]['color'])
        return node

    def setSettings(self, settings):
        """
        restore graph architecture and node parameters

        Parameters
        ----------
        settings: dict
            the dict-like description of the graph

        """
        for name, values in settings.items():
            node = self.addNode(values['state']['type'], values['state']['parents'], values['state']['position'])
            node.setSettings(values)

    def getSettings(self):
        """
        get graph architecture and node parameters

        Return
        ------
        settings: dict

        """
        settings = {}
        orderedNodes = []
        nodes = list(self.nodes.values())
        while nodes:
            node = nodes.pop(0)
            if not node.parents or set(node.parents).intersection(orderedNodes):
                settings[node.name] = node.getSettings()
                orderedNodes.append(node)
            else:
                nodes.append(node)
        return settings
