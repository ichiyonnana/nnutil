# coding:utf-8
"""
UI 操作周りのヘルパー
"""


import re

import pymel.core as pm

window_width = 300
header_width = 50
color_x = (1.0, 0.5, 0.5)
color_y = (0.5, 1.0, 0.5)
color_z = (0.5, 0.5, 1.0)
color_joint = (0.5, 1.0, 0.75)
color_select = (0.5, 0.75, 1.0)
bw_single = 24
bw_double = bw_single*2 + 2

button_width_auto = -1
button_width1 = 24
button_width2 = button_width1*2 + 2
button_width3 = button_width1*3 + 4
button_width4 = button_width1*4 + 6


handle_method = {
    "floatField": pm.floatField,
    "intField": pm.intField,
    "checkBox": pm.checkBox,
    "button": pm.button,
}


def any_handler(*args):
    pass


def get_component_type(commponent):
    return re.match(r"^.*\|(.*?)\d*", component).groups()[0]


def decide_width(word):
    actual_width = 0

    if len(word) <= 2:
        actual_width = button_width1
    elif len(word) <= 8:
        actual_width = button_width2
    elif len(word) < 14:
        actual_width = button_width3
    else:
        actual_width = button_width4

    return actual_width


def column_layout():
    return pm.columnLayout()


def row_layout(numberOfColumns=16):
    return pm.rowLayout(numberOfColumns=numberOfColumns)


def end_layout():
    pm.setParent("..")


def header(label):
    return pm.text(label=label, width=header_width)


def text(label="", width=button_width_auto):
    actual_width = width

    if width == button_width_auto:
        actual_width = decide_width(label)

    return pm.text(label=label, width=actual_width)


def button(label, width=button_width_auto, c="", dgc=""):
    actual_width = width

    if width == button_width_auto:
        actual_width = decide_width(label)

    component = pm.button(l=label, c=c, dgc=dgc, width=actual_width)

    return component


def float(v=0, en=True, dc="", width=bw_double):
    return pm.floatField(v=v, en=en, dc=dc, width=bw_double)


def separator(width=window_width):
    return pm.separator(width=window_width)


def get_value(component):
    if "floatField" in component:
        return pm.floatField(component, q=True, v=True)

    elif "intField" in component:
        return pm.intField(component, q=True, v=True)

    elif "checkBox" in component:
        return pm.checkBox(component, q=True, v=True)

    else:
        raise


def set_value(component, value):
    if "floatField" in component:
        return pm.floatField(component, e=True, v=value)

    elif "intField" in component:
        return pm.intField(component, e=True, v=int(value))

    elif "checkBox" in component:
        return pm.checkBox(component, e=True, v=bool(value))
    
    else:
        raise


def check_box(label="", v=False, cc=None):
    return pm.checkBox(label=label, v=v, cc=cc)


def set_availability(component, stat):
    component_type = get_component_type(component)

    if component_type in handle_method.keys():
        handle_method[component_type](component, e=True, en=True)
    else:
        raise


def enable_ui(component):
    set_availability(component, True)


def disable_ui(component):
    set_availability(component, False)


