import argparse
from genericpath import isdir
import importlib
import os


from .modules.abstract_attention import AbstractAttention

MODULE_REGISTRY = {}


# from https://stackoverflow.com/a/49753634
def remove_argument(parser, arg):
    for action in parser._actions:
        opts = action.option_strings
        if (opts and opts[0] == arg) or action.dest == arg:
            parser._remove_action(action)
            break

    for action in parser._action_groups:
        for group_action in action._group_actions:
            if group_action.dest == arg:
                action._group_actions.remove(group_action)
                return

# a wrapper of add_argument() method that handles dest variable automatically.
def add_nested_argument(parser, name, **kwargs):
    parser.add_argument(name, dest='attn_args.{}'.format(name.lstrip('-').replace('-', '_')), **kwargs)

# copied from https://stackoverflow.com/a/18709860;
# a useful variant that supports nested arg parsing!
class NestedNamespace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, NestedNamespace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

def register_cls(cls):
    """
    Register a class with its name

    Args:
        cls: a new class fro registration
    """
    name = cls.__name__
    if name in MODULE_REGISTRY:
        raise ValueError(f'Cannot register duplicate class ({name})')
    if not issubclass(cls, AbstractAttention):
        raise ValueError(f'Class {name} must extend AbstractAttention')
    if name in MODULE_REGISTRY:
        raise ValueError(f'Cannot register class with duplicate class name ({name})')
    MODULE_REGISTRY[name] = cls
    return cls


from .modules.multihead_attention import MultiheadAttention 
from .modules.abc import ABC
from .modules.cosformer import CosformerAttention
from .modules.lara import Lara
from .modules.local_attention import LocalAttention
from .modules.nystrom_attention import NystromAttention
from .modules.performer import Performer
from .modules.s4d import S4D
from .modules.transformer_ls import AttentionLS, CausalLS
from .modules.probsparse import ProbSparse


def get_cls(name):
    """
    Create a class with configuration

    Args:
        name: configuration dictionary for building class

    Returns:
        - an instance of class
    """
    return MODULE_REGISTRY[name]


AVAILABLE_ATTENTIONS = [_ for _ in MODULE_REGISTRY]



