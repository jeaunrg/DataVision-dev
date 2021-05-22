PyQTapp_template


HOW TO ADD A NEW MODULE:

- create a widget ModuleName.ui with QtDesigner and save it in resources/design/ui
    this widget should contain at least a QLabel named 'state' and a QPorgressBar named 'loading'
    if you need to see the process progression.

- In the Model, create the method that will do the raw process of your module
    add the decorator 'protector' to this method.

- In the Presenter method init_module_connections(), add a new condition (if module_name == "ModuleName":)
    to initialize and connect the buttons of your module.
    Buttons must be connected to Presenter methods.
    The presenter methods will get all the widget information and will call the appropriate
    model method (with raw arguments).
