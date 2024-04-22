# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2021-08-16 16:03:17
# @Last Modified by:   Shilong Liu
# @Last Modified time: 2022-01-23 15:26
# modified from mmcv

import inspect
from functools import partial


# 可以将Register视为一个`dict`
# 用于实现注册机制。它允许将函数或对象注册到一个字典中，并通过键进行查找和获取。
class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    def __len__(self):
        return len(self._module_dict)

    # @property 修饰的 name 方法将作为 Registry 类的一个属性，用于返回注册表的名称。
    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def registe_with_name(self, module_name=None, force=False):
        return partial(self.register, module_name=module_name, force=force)

    # 定义一个模型文件，通过register注册到_module_dict字典中，通过key自动查询到模型
    def register(self, module_build_function, module_name=None, force=False):
        """Register a module build function.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isfunction(module_build_function):
            raise TypeError('module_build_function must be a function, but got {}'.format(
                type(module_build_function)))
        if module_name is None:
            module_name = module_build_function.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_build_function

        return module_build_function


MODULE_BUILD_FUNCS = Registry('model build functions')
