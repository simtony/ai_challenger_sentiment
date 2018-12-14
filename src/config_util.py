import re
import argparse
import logging
import types
from tabulate import tabulate
logger = logging.getLogger(__name__)


class AttributeListAppender(object):
    def __init__(self, convertible, list_name):
        if not isinstance(convertible, Convertible):
            raise TypeError("Expect object as type {}, got {}: {}".format(Convertible,
                                                                          type(convertible),
                                                                          convertible))
        if not hasattr(convertible, '_param_types'):
            raise ValueError("Attribute '_param_types' not defined in {}".format(type(convertible)))

        if list_name not in convertible._param_types:
            convertible._param_types[list_name] = []
        elif not isinstance(convertible._param_types[list_name], list):
            raise TypeError("Expect a {} for '{}._param_types[{}]', got {}.".format(list,
                                                                                    type(convertible),
                                                                                    list_name,
                                                                                    type(convertible._param_types[
                                                                                             list_name])))
        self.convertible = convertible
        self.list_name = list_name

        self.attribute_list = convertible._param_types[list_name]
        self.old_setattr = self.convertible.__class__.__setattr__

    def __enter__(self):
        # intercept setter to append the name of attribute to a list
        # name starts with "_" will not be affected
        old_setattr = self.old_setattr

        def new_setattr(obj, name, value):
            old_setattr(obj, name, value)
            # avoid duplication and internal attributes started with '_'
            if name not in self.attribute_list and not re.match('_', name):
                self.attribute_list.append(name)
                logger.debug("Add '{}' as type '{}'.".format(name, self.list_name))

        self.convertible.__class__.__setattr__ = new_setattr

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.convertible.__class__.__setattr__ = self.old_setattr


class setups(AttributeListAppender):
    def __init__(self, convertible):
        super(setups, self).__init__(convertible, 'setups')


class learning(AttributeListAppender):
    def __init__(self, convertible):
        super(learning, self).__init__(convertible, 'learning')


class structures(AttributeListAppender):
    def __init__(self, convertible):
        super(structures, self).__init__(convertible, 'structures')


class directories(AttributeListAppender):
    def __init__(self, convertible):
        super(directories, self).__init__(convertible, 'directories')


class immutables(AttributeListAppender):
    def __init__(self, convertible):
        super(immutables, self).__init__(convertible, 'immutables')


class FuncMap(dict):
    def __init__(self, default_map, *args, **kwargs):
        super(FuncMap, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v
        object.__setattr__(self, "default_map", default_map)

    def __getattr__(self, key):
        func = self.get(key)
        if func is None:
            if self.default_map is None or self.default_map.get(key) is None:
                raise ValueError("Function {} not defined and no default.".format(key))
            else:
                logger.debug("Function {} not defined, return default version.".format(key))
                return self.default_map.get(key)
        else:
            return func

    def __getitem__(self, key):
        super(FuncMap, self).__getitem__(key)

    def __setattr__(self, key, func):
        self.__setitem__(key, func)

    def __setitem__(self, key, func):
        super(FuncMap, self).__setitem__(key, func)
        self.__dict__.update({key: func})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(FuncMap, self).__delitem__(key)
        del self.__dict__[key]


class Convertible(object):
    def __init__(self):
        object.__setattr__(self, "_dependency_dict", {})  # dependency dict for chained attribute updates
        object.__setattr__(self, "_argparse_types", (int, float, str, bool))  # what types can argparse change
        object.__setattr__(self, "_param_types", {})  # attribute list for use-defined types
        object.__setattr__(self, "_structs", {})  # {struct_name -> value -> func_name -> func}

    def log_params(self):
        s = "model params: \n"
        for attr_name, attr_list in self._param_types.items():
            if attr_name == "immutables":
                continue
            s += attr_name + ':\n'
            s += tabulate([(attr, repr(self.__getattribute__(attr))) for attr in attr_list
                           if isinstance(self.__getattribute__(attr), self._argparse_types)], disable_numparse=True)
            s += '\n\n'
        s = s.replace('\n', '\n\t')
        logger.info(s)

    def struct_def(self, struct_name, value, is_default=False):
        if struct_name not in self.__dict__:
            with structures(self):
                logger.warning(
                        "'{}' not defined in {}, auto generate with value: {}.".format(struct_name, self.__class__,
                                                                                       value))
                self.__setattr__(struct_name, value)

        if struct_name not in self._structs:
            self._structs[struct_name] = {None: FuncMap(default_map=None)}
            self._structs[struct_name][value] = FuncMap(default_map=self._structs[struct_name][None])

        elif value not in self._structs[struct_name]:
            self._structs[struct_name][value] = FuncMap(default_map=self._structs[struct_name][None])

        def decorator(func):
            # self._structs[struct_name][None], self._structs[struct_name][value] is always available
            func_name = func.__name__
            logger.info("Structural param: {}={}, add function: '{}'".format(struct_name, value, func_name))
            # test argcount
            for func_map in self._structs[struct_name].values():
                if func_name in func_map:
                    argcount = func_map.__getattr__(func_name).__code__.co_argcount
                    if func.__code__.co_argcount != argcount:
                        raise ValueError(
                                "Function '{}' of structural param {}={} should have {} args, got {} args.".format(
                                        func_name, struct_name, value, argcount, func.__code__.co_argcount))
                    break
            # set map
            self._structs[struct_name][value][func_name] = func
            # set default
            if is_default:
                self._structs[struct_name][None][func_name] = func
            return func

        return decorator

    def struct(self, struct_name):
        return self._structs[struct_name][self.__getattribute__(struct_name)]

    def __setattr__(self, name, value):
        if name in self.__dict__ and value == object.__getattribute__(self, name):
            # do nothing when assigning the same value to the attribute
            return

        object.__setattr__(self, name, value)
        if isinstance(value, types.LambdaType):
            if name not in self.__dict__:
                # lazy initialization
                object.__setattr__(self, '_' + name, None)
                object.__setattr__(self, '_' + name + '_to_update', True)
            else:
                object.__setattr__(self, '_' + name, value())
                object.__setattr__(self, '_' + name + '_to_update', False)
            # co_names is every name referenced in lambda declaration.
            # when dependent changes, the cache of this attribute _name should be updated
            for parent in value.__code__.co_names:
                # may introduce unnecessary updates since co_names may accidentally be in self.__dict__
                if parent in self.__dict__:
                    if parent in self._dependency_dict:
                        self._dependency_dict[parent].append(name)
                    else:
                        self._dependency_dict[parent] = [name]
        else:
            if name in self._dependency_dict:
                for child in self._dependency_dict[name]:
                    object.__setattr__(self, '_' + child + '_to_update', True)

    def __getattribute__(self, name):
        # intercepts all object.attribute operators
        # if attribute is a function, return the value after call the function
        value = object.__getattribute__(self, name)
        if isinstance(value, types.LambdaType):
            if object.__getattribute__(self, '_' + name + '_to_update'):
                self.__setattr__('_' + name, value())
                self.__setattr__('_' + name + '_update', False)
            value = object.__getattribute__(self, '_' + name)
        return value

    def to_parser(self):
        """
        Convert all params into args of argument parser.

        Returns:
            parser: object of argparse.ArgumentParser
        """

        # helper function for bool type
        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError("Expect value in ['yes'/'no', 'true'/'false', "
                                                 "'t'/'f', 'y'/'n', '1'/'0'] for bool attribute, got {0}".format(
                        repr(v)))

        parser = argparse.ArgumentParser(add_help=False)

        for name in self.__dict__.keys():
            if re.match('_', name) is not None:
                continue
            if 'immutables' in self._param_types and name in self._param_types["immutables"]:
                continue
            if type(getattr(self, name)) not in self._argparse_types:
                continue

            dtype = type(getattr(self, name))
            if dtype == bool:
                parser.add_argument("--{}".format(name), type=str2bool,
                                    metavar='<T/F: {}>'.format(repr(self.__getattribute__(name))))
            else:
                parser.add_argument("--{}".format(name), type=dtype,
                                    metavar='<{}: {}>'.format(str(dtype.__name__), repr(self.__getattribute__(name))))
        return parser

    # helper functions
    def _diff_keys(self, update_dict, host_keys=None):
        if host_keys is None:
            host_keys = self.__dict__.keys()
        update_keys = update_dict.keys()

        host_keys = set(host_keys)
        update_keys = set(update_keys)

        host_uniq_keys = list(host_keys - update_keys)
        host_uniq_keys.sort()
        update_uniq_keys = list(update_keys - host_keys)
        update_uniq_keys.sort()

        common_keys = host_keys & update_keys
        value_diff_keys = [key for key in common_keys if getattr(self, key) != update_dict[key]]
        value_diff_keys.sort()
        return host_uniq_keys, update_uniq_keys, value_diff_keys

    def _update_on_diff_keys(self, update_dict, diff_keys):
        for key in diff_keys:
            logger.info("Update '{0}' from {1} to {2}".format(key, repr(getattr(self, key)), repr(update_dict[key])))
            setattr(self, key, update_dict[key])

    def from_args(self, args):
        """
        Update params from argparse results.

        Args:
            args: argparse reuslts

        """
        logger.info('Load argparse results.')
        # filter out unset args
        update_dict = dict([(key, value) for key, value in vars(args).items() if value is not None])
        _, _, diff_keys = self._diff_keys(update_dict)
        if diff_keys:
            self._update_on_diff_keys(update_dict, diff_keys)
        else:
            logger.info('No params to be updated from args.')


if __name__ == '__main__':
    import os

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)


    class Test(Convertible):
        def __init__(self):
            super(Test, self).__init__()
            with structures(self):
                self.param_num = 0
                self.param_str = '666'
                self.param_float = 1.0
                self.param_bool = False
                self.param_list = [1, 2, 3]
                self._param_private = 233
                self.param_lambda_num = lambda: self.param_num + 1
                self.param_lambda_dir = lambda: os.path.join('path', self.param_str)
                with directories(self):
                    self.param_dir_num = 0
                    # chained lambda
                    self.param_dir_lambda_dir = lambda: os.path.join(self.param_lambda_dir, self.param_str)
            with directories(self):
                self.dir_num = 10


    test = Test()
    # types
    print test._param_types

    # getter
    print 'param_num', test.param_num
    print 'param_str', test.param_str
    print 'param_float', test.param_float
    print 'param_bool', test.param_bool
    print 'param_list', test.param_list
    print '_param_private', test._param_private
    print 'param_lambda_num', test.param_lambda_num
    print 'param_lambda_dir', test.param_lambda_dir
    print 'param_dir_num', test.param_dir_num
    print 'param_dir_lambda_dir', test.param_dir_lambda_dir

    # setter
    test.param_num = 1
    test.param_str = '777'
    test.param_float = 2.0
    test.param_bool = True
    test.param_list = [3, 3, 3]
    test._param_private = 666
    test.param_lambda_num = 13

    print '\nchanged attrs'
    print 'param_num', test.param_num
    print 'param_str', test.param_str
    print 'param_float', test.param_float
    print 'param_bool', test.param_bool
    print 'param_list', test.param_list
    print '_param_private', test._param_private
    print 'param_lambda_num', test.param_lambda_num
    print 'param_lambda_dir', test.param_lambda_dir
    print 'param_dir_num', test.param_dir_num
    print 'param_dir_lambda_dir', test.param_dir_lambda_dir


    # struct_def
    @test.struct_def('block', False)
    def inference(a, b):
        return a + b


    @test.struct_def('block', True)
    def inference(a, b):
        return a * b


    @test.struct_def('block', True, is_default=True)
    def p():
        return 0


    @test.struct_def('block', True, is_default=True)
    def p2():
        return 1


    print "block = True"
    test.block = True
    print test.struct('block').inference(100, 200)
    print test.struct('block').p()
    print test.struct('block').p2()

    print "block = False"
    test.block = False
    print test.struct('block').inference(100, 200)
    print test.struct('block').p()
    print test.struct('block').p2()

    # argparse
    parser = test.to_parser()
    parser.print_usage()
    # parser.print_help()
