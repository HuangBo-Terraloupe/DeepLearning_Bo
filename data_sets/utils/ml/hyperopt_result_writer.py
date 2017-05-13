import collections
import os

from hyperopt import pyll
from hyperopt.base import STATUS_OK
from objectdetection.utils.config import load_config

from objectdetection.utils.s3_client import S3Client

config = load_config()


def nested_update(target, source):
    """Deep update of python directories."""
    for k, v in source.iteritems():
        if isinstance(v, collections.Mapping):
            subdict = nested_update(target.get(k, {}), v)
            target[k] = subdict
        else:
            target[k] = source[k]
    return target


class ResultWriter:
    def __init__(self, ctrl, save_mongo=True, s3_bucket=config.s3_bucket):
        self.ctrl = ctrl
        self.save_mongo = save_mongo
        self.result = ctrl.current_trial["result"]
        self.s3_bucket = s3_bucket

    def update(self, dct):
        """Nested update of result dict.

        Args:
            dct (dict): Dictionary to merge.

        """
        result = self.result
        nested_update(result, dct)
        self.__save()

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getitem__(self, key):
        return self.set(key)

    def __delitem__(self, key):
        del self.result[key]
        self.__save()

    def get(self, path=None):
        """Get value at specific key.

        Args:
            path (string): Lookup key, can be in path format (e.g. 'foo.bar')
                           to look up in a nested dict.
        Return:
            value at the given position.
        """
        if not path:
            return self.result
        keys = path.split(".")
        return self.__get_by_keys(keys)

    @property
    def exp_key(self):
        return self.ctrl.current_trial["exp_key"]

    @property
    def job_id(self):
        return str(self.ctrl.current_trial["_id"])

    def set(self, path, value):
        """Set value at specific key.

        Args:
            path (string): Key, can be in path format (e.g. 'foo.bar')
            value:         Value to save, can be any type
        """
        keys = path.split(".")
        subdict = self.__get_by_keys(keys[:-1])
        subdict[keys[-1]] = value
        self.__save()

    @property
    def s3_prefix(self):
        return os.path.join(config.s3_prefix, self.exp_key, self.job_id)

    def upload_result_file(self, filename):
        client = S3Client(bucket=self.s3_bucket, prefix=self.s3_prefix)
        return client.upload_file(filename)

    def append(self, path, value):
        """Append value to an array at given position.

        If the value at the given position is an array, append to it.
        If the key does not exist, create array containing the given value
        Else, error.

        Args:
            path (string): Key, can be in path format (e.g. 'foo.bar')
            value:         Value to append, can be any type
        """
        keys = path.split(".")
        subdict = self.__get_by_keys(keys[:-1])
        last_key = keys[-1]
        if last_key not in subdict:
            subdict[last_key] = [value]
        elif isinstance(subdict[last_key], list):
            subdict[last_key].append(value)
        else:
            raise RuntimeError("Value at '%s' is not a list" % path)
        self.__save()

    def __get_by_keys(self, keys):
        current = self.result
        for k in keys:
            current = current.get(k)
        return current

    def __save(self):
        if self.save_mongo:
            self.ctrl.checkpoint(self.result)


class WithResultWriter(object):
    """A decorator class for realtime hyperopt objectives.

    Usage:
        objective = WithResultWriter(objective)
    """

    def __init__(self, objective, save_arguments=False):
        """Constructor
        Args:
            objective (function): The objective to minimize.
                             Expects a function with signature ((config, writer) -> loss),

                             where `writer` is an instance of ResultWriter, that can be used to
                             write values to mongodb at runtime, `config` is the configuration
                             point in parameter space, suggested by hyperopt.

                             e.g:

                             def objective(x, result_writer):
                                 for i in xrange(10)
                                     result_writer.append("values", i)
                                 return x ** 2
            save_arguments (boolean): Whether to current hyper parameter set to the results.
        """
        self.objective = objective
        self.save_arguments = save_arguments
        self.fmin_pass_expr_memo_ctrl = True

    def __call__(self, expr, memo, ctrl):
        result_writer = ResultWriter(ctrl)

        config = pyll.rec_eval(expr, memo=memo)
        if self.save_arguments:
            result_writer.set("arguments", config)
        loss = self.objective(config, result_writer)
        result_writer.update({'status': STATUS_OK, 'loss': loss})
        return result_writer.result
