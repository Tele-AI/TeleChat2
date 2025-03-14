# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" Transformer-Config dict parse module """

import argparse
from argparse import Action
import copy
import os
from collections import OrderedDict
from typing import Union

import yaml
from mindformers.tools.check_rules import check_yaml_depth_before_loading
from .template import ConfigTemplate

BASE_CONFIG = 'base_config'


class DictConfig(dict):
    """config"""
    def __init__(self, **kwargs):
        super(DictConfig, self).__init__()
        self.update(kwargs)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def __deepcopy__(self, memo=None):
        """Deep copy operation on arbitrary MindFormerConfig objects.

        Args:
            memo (dict) : Objects that already copied.
        Returns:
            MindFormerConfig : The deep copy of the given MindFormerConfig object.
        """
        config = self.__class__()
        for key in self.keys():
            config.__setattr__(copy.deepcopy(key, memo),
                               copy.deepcopy(self.__getattr__(key), memo))
        return config

    def to_dict(self):
        """
        for yaml dump,
        transform from Config to a strict dict class
        """
        return_dict = {}
        for key, val in self.items():
            if isinstance(val, self.__class__):
                val = val.to_dict()
            return_dict[key] = val
        return return_dict


class MindFormerConfig(DictConfig):
    """
    A class for configuration that inherits from Python's dict class.
    Can parse configuration parameters from yaml files or dict instances.

    Args:
        args (Any): Extensible parameter list, either a yaml configuration file path or a configuration dictionary.
        kwargs (Any): Extensible parameter dictionary, either a yaml configuration file path or a configuration
            dictionary.

    Returns:
        An instance of the class.

    Examples:
        >>> from mindformers.tools import MindFormerConfig
        >>>
        >>> # test.yaml:
        >>> #     a:1
        >>> cfg = MindFormerConfig('./test.yaml')
        >>> cfg.a
        1
        >>> cfg = MindFormerConfig(**dict(a=1, b=dict(c=[0,1])))
        >>> cfg.b
        {'c': [0, 1]}
    """

    def __init__(self, *args, **kwargs):
        super(MindFormerConfig, self).__init__()
        cfg_dict = {}

        # load from file
        load_from_file = False
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('yaml') or arg.endswith('yml'):
                    raw_dict = MindFormerConfig._file2dict(arg)
                    cfg_dict.update(raw_dict)
                    load_from_file = True

        # load dictionary configs
        if kwargs is not None:
            cfg_dict.update(kwargs)

        if load_from_file:
            ConfigTemplate.apply_template(cfg_dict)
        MindFormerConfig._dict2config(self, cfg_dict)

    def merge_from_dict(self, options):
        """
        Merge options into config.

        Args:
            options (dict): Configuration options that need to be merged.

        Examples:
            >>> from mindformers.tools import MindFormerConfig
            >>>
            >>> options = {'model.arch': 'LlamaForCausalLM'}
            >>> cfg = MindFormerConfig(**dict(model=dict(model_config=dict(type='LlamaConfig'))))
            >>> cfg.merge_from_dict(options)
            >>> print(cfg)
            {'model': {'model_config': {'type': 'LlamaConfig'}, 'arch': 'LlamaForCausalLM'}}
        """
        option_cfg_dict = {}
        for full_key, value in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for sub_key in key_list[:-1]:
                d.setdefault(sub_key, MindFormerConfig())
                d = d.get(sub_key)
            sub_key = key_list[-1]
            d[sub_key] = value
        merge_dict = MindFormerConfig._merge_a_into_b(option_cfg_dict, self)
        MindFormerConfig._dict2config(self, merge_dict)

    @staticmethod
    def _merge_a_into_b(a, b):
        """Merge dict ``a`` into dict ``b``

        Values in ``a`` will overwrite ``b``

        Args:
            a (dict) : The source dict to be merged into b.
            b (dict) : The origin dict to be fetch keys from ``a``.
        Returns:
            dict: The modified dict of ``b`` using ``a``.
        """
        b = b.copy() if b is not None else a.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                b[k] = MindFormerConfig._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def _file2dict(filename=None):
        """Convert config file to dictionary.

        Args:
            filename (str) : config file.
        """
        if filename is None:
            raise NameError('This {} cannot be empty.'.format(filename))

        filepath = os.path.realpath(filename)
        with open(filepath, encoding='utf-8') as fp:
            check_yaml_depth_before_loading(fp)
            fp.seek(0)
            cfg_dict = yaml.safe_load(fp)
            cfg_dict = OrderedDict(sorted(cfg_dict.items()))

        # Load base config file.
        if BASE_CONFIG in cfg_dict:
            cfg_dir = os.path.dirname(filename)
            base_filenames = cfg_dict.pop(BASE_CONFIG)
            base_filenames = base_filenames if isinstance(
                base_filenames, list) else [base_filenames]

            cfg_dict_list = list()
            for base_filename in base_filenames:
                cfg_dict_item = MindFormerConfig._file2dict(
                    os.path.join(cfg_dir, base_filename))
                cfg_dict_list.append(cfg_dict_item)

            base_cfg_dict = dict()
            for cfg in cfg_dict_list:
                base_cfg_dict.update(cfg)

            # Merge config
            base_cfg_dict = MindFormerConfig._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

        return cfg_dict

    @staticmethod
    def _dict2config(config, dic):
        """Convert dictionary to config.

        Args:
            config : Config object
            dic (dict) : dictionary
        Returns:

        Exceptions:

        """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = MindFormerConfig()
                    dict.__setitem__(config, key, sub_config)
                    MindFormerConfig._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]


    def get_value(self, levels: Union[str, list], default=None):
        """Get the attribute according to levels, if not exist return default.

        Args:
            levels : the level to be accessed
            default : None
        Returns:
            default or value of the key to be accessed

        Examples:
        >>> config = MindFormerConfig(**{'context': {'mode': 'GRAPH_MODE'}, 'parallel': {}})
        >>> config.get_value(['context', 'mode'])
        >>> 'GRAPH_MODE'
        >>> config.get_value(['context', 'mode'], 'DEFAULT_MODE')
        >>> 'GRAPH_MODE'
        >>> config.get_value(['context', 'fake_mode'])
        >>> None
        >>> config.get_value(['context', 'fake_mode'], 'DEFAULT_MODE')
        >>> 'DEFAULT_MODE'
        >>> config.get_value('context.mode', 'DEFAULT_MODE')
        >>> 'GRAPH_MODE'
        >>> config.get_value('context.fake_mode', 'DEFAULT_MODE')
        >>> 'DEFAULT_MODE'
        """

        if not levels:
            return default
        if isinstance(levels, str):
            levels = levels.split('.')
        if len(levels) == 1:
            config = self or {}
            return config.get(str(levels[-1]), default)
        if getattr(self, levels[0]):
            return getattr(self, levels[0]).get_value(
                levels[1:], default
            )
        return default

    def set_value(self, levels: Union[list, str], value):
        """set the attribute according to levels.

        Args:
            levels : the level to be accessed
            value : The value to be set
        Returns:
            None

        Examples:
        >>> config = MindFormerConfig(**{'context': {'mode': 0}, 'parallel': {}, 'test': None})
        >>> config.set_value('context.mode', 1)
        >>> config = {'context': {'mode': 1}, 'parallel': {}, 'test': None})
        >>> config.set_value(['context', 'device_id'], 2)
        >>> config = {'context': {'mode': 1, device_id: 2}, 'parallel': {}, 'test': None})
        >>> config.set_value('parallel', {'hello', 'mf'})
        >>> config = {'context': {'mode': 1, device_id: 2}, 'parallel': {'hello', 'mf'}, 'test': None})
        >>> config.set_value('test.model', 1)
        >>> config = {'context': {'mode': 1, device_id: 2}, 'parallel': {'hello', 'mf'}, 'test': {'model': 1}}})
        >>> config.set_value('test', {'data', 8)
        >>> config = {'context': {'mode': 1, device_id: 2}, 'parallel': {'hello', 'mf'}, 'test': {'data': 8}}})
        """
        if levels:
            if isinstance(levels, str):
                levels = levels.split('.')
            if len(levels) == 1:
                if isinstance(self, MindFormerConfig):
                    setattr(self, levels[-1], value)
                return
            config = getattr(self, levels[0]) or MindFormerConfig()
            setattr(self, levels[0], config)
            self.get(levels[0]).set_value(levels[1:], value)


class ActionDict(Action):
    """
    Argparse action to split an option into KEY=VALUE from on the first =
    and append to dictionary.
    List options can be passed as comma separated values.
    i.e. 'KEY=Val1,Val2,Val3' or with explicit brackets
    i.e. 'KEY=[Val1,Val2,Val3]'.
    """

    @staticmethod
    def _parse_int_float_bool(val):
        """convert string val to int or float or bool or do nothing."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.upper() in ['TRUE', 'FALSE']:
            return val.upper == 'TRUE'
        return val

    @staticmethod
    def find_next_comma(val_str):
        """find the position of next comma in the string.

        note:
            '(' and ')' or '[' and']' must appear in pairs or not exist.
        """
        if val_str.count('(') != val_str.count(')') or \
               (val_str.count('[') != val_str.count(']')):
            raise ValueError("( and ) or [ and ] must appear in pairs or not exist.")

        end = len(val_str)
        for idx, char in enumerate(val_str):
            pre = val_str[:idx]
            if ((char == ',') and (pre.count('(') == pre.count(')'))
                    and (pre.count('[') == pre.count(']'))):
                end = idx
                break
        return end

    @staticmethod
    def _parse_value_iter(val):
        """Convert string format as list or tuple to python list object
        or tuple object.

        Args:
            val (str) : Value String

        Returns:
            list or tuple

        Examples:
            >>> ActionDict._parse_value_iter('1,2,3')
            [1,2,3]
            >>> ActionDict._parse_value_iter('[1,2,3]')
            [1,2,3]
            >>> ActionDict._parse_value_iter('(1,2,3)')
            (1,2,3)
            >>> ActionDict._parse_value_iter('[1,[1,2],(1,2,3)')
            [1, [1, 2], (1, 2, 3)]
        """
        # strip ' and " and delete whitespace
        val = val.strip('\'\"').replace(" ", "")

        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            # remove start '(' and end ')'
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            # remove start '[' and end ']'
            val = val[1:-1]
        elif ',' not in val:
            return ActionDict._parse_int_float_bool(val)

        values = []
        len_of_val = len(val)
        while len_of_val > 0:
            comma_idx = ActionDict.find_next_comma(val)
            ele = ActionDict._parse_value_iter(val[:comma_idx])
            values.append(ele)
            val = val[comma_idx + 1:]
            len_of_val = len(val)

        if is_tuple:
            return tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for key_value in values:
            key, value = key_value.split('=', maxsplit=1)
            options[key] = self._parse_value_iter(value)
        setattr(namespace, self.dest, options)


def ordered_yaml_dump(data, stream=None, yaml_dumper=yaml.SafeDumper,
                      object_pairs_hook=OrderedDict, **kwargs):
    """Dump Dict to Yaml File in Orderedly."""
    class OrderedDumper(yaml_dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())

    OrderedDumper.add_representer(object_pairs_hook, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwargs)


def parse_args():
    """
    Parse arguments from `yaml or yml` config file.

    Returns:
        object: argparse object.
    """
    parser = argparse.ArgumentParser("Transformer Config.")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="",
                        help='Enter the path of the model config file.')

    return parser.parse_args()
