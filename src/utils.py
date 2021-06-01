from datetime import timedelta


def empty_to_none(arg):
    if not arg:
        return None
    else:
        return arg


def ceval(arg):
    try:
        return eval(arg)
    except (NameError, SyntaxError):
        return empty_to_none(arg)


# def protector(foo):
#     """
#     function used as decorator to avoid the app to crash because of basic errors
#     """
#     def inner(*args, **kwargs):
#         try:
#             return foo(*args, **kwargs)
#         except Exception as e:
#             print(e)
#             return e
#     return inner
