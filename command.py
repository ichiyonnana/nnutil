#! python
# coding:utf-8

import maya.cmds as cmds
import maya.mel as mel

def get_selection():
    return cmds.ls(selection=True, flatten=True)