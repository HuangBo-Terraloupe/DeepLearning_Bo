import click


def flatten_dict(input_dict, parents=None, inter_results=None):
    """ Flattens a dictionary

    If the leaf is a dict, it will remain as value

    Args:
        input_dict: input dict
        parents: Transitional recursion parameter
        inter_results: Transitional recursion parameter

    Returns:
        A  list of tuples (path, value), where path is a list of keys

    Examples:
        >>> flatten_dict({'foo': {'bar': 1}}) == [(['foo'], {'bar': 1})]
        >>> flatten_dict({'foo': {'bar': {'spam': 1}}}) == [(['foo', 'bar'], {'spam': 1})]
    """
    parents = parents or []
    results = inter_results or []

    for key, value in input_dict.iteritems():
        path = parents + [key]
        if type(value) == dict and type(value.values()[0]) == dict:
            results = flatten_dict(value, parents=path, inter_results=results)
        else:
            results = results + [(path, value)]

    return results


def get_path(obj, path=None):
    """ Get value by path in dict

    Args:
        obj (dict): Dictionary
        path (list): List of nested keys to look up

    Return:
        Leaf value, or None if not exists
    """
    if not path:
        return obj

    for k in path:
        if k in obj:
            return get_path(obj[k], path[1:len(path)])
        else:
            return None
#
# Defines utility class for handling metadata schema
#


class MetaSchema:
    """ Class for handling meta schema, i.e. serializing and deserializing meta options from click, accordning to a nested structure.

        Args:
            meta_schema (dict): Dictionary of the metadata schemas.
                              Arbitrary nested dict, where the leaves are,
                              with a schema dict at each leave, which
                              may consists of the fields:
                              type, description, required, enum, default


        Example:

            schema_def = {"image": {
                             "format": {
                                 "type": str,
                                 "description": "Image format",
                                 "required": False,
                                 "default": "tif",
                                 "enum": ["tif", "jpg"],
                             },
                             "file": {
                                 "type": file,
                                 "exists": True,
                                 "required": True
                             }
                           }}

            schema = MetaSchema(schema_def)

            @click.command()
            @schema.click_options("image_format", "image_file")
            def main(**options):

                # create meta data dict, where attributes are filled in the following order:
                # None, default values, click options
                metadata = schema.create_meta(options)

    """

    def __init__(self, meta_schema):
        self._meta_schema = meta_schema
        self._flat_meta_schema = self.flatten(meta_schema)

    def click_options(self, *options):
        """ Decorator to apply click options

            Args:
                flat option keys, that should be considered to create click option

            Exceptions:

                - when the options is not found in the meta data definition.
                  Beware, that options are flat values, i.e. underscore-joined
                  keys of the nested attributes

            Example: See above for full usage example!
        """

        def func_wrapper(func):
            for args, kwargs in self._click_option_args(options):
                click_wrapper = click.option(*args, **kwargs)
                func = click_wrapper(func)
            return func

        return func_wrapper

    def flatten(self, schema):
        """ Flatten a nested schema by combining keys:

            Example:

                input: {foo: {bar: {<schema>}}}
                output: {foo_bar: {<schema>}}
        """

        flat_schema = flatten_dict(schema)
        return {"_".join(keys): value for keys, value in flat_schema}

    def create_meta(self, options={}):
        """ Generate nested meta dict according to the schema.

            Default values are applied, if specified, but may be overwritten by options.
            Types and enums are currently not checked.

            Arguments:
                options (dict): Additional flat options (in snake case) to be merge in
                                the meta data (e.g. from a click function)


            Args:
                options (dict): option dict, with keys in snake case
                                (e.g. from a click function)

            Return:
                Nested dict, where values are are filled from options if present,
                or with None otherwise.

            Example:

                schema: {"image": {"format": {<schema>}}, {"type": {<schema>}}}
                options: {"image_format": "tif"}
                return: {"image_format": "tif", "image_type": None }
        """

        return self._create_meta(options, self._meta_schema)

    def validate(self, data, raise_on_error=True):
        """ Validate a python dict according to the schema.

        Args:
            data (dict): Python dict to validate
        """
        assert isinstance(data, dict)

        missing, invalid_types = self._validate(data)
        is_valid = (not missing) and (not invalid_types)
        if raise_on_error and not is_valid:
            raise ValidationError(
                self._build_validation_msg(missing, invalid_types))

        return is_valid

    def _build_validation_msg(self, missing, invalid_types):
        def join_path(path):
            return ".".join(path)

        def format_type(type_spec):
            # enum
            if isinstance(type_spec, list):
                return "one of {}".format(", ".join(type_spec))
            # other type
            return type_spec.__name__

        msg = "Schema specification not met"
        msg += "\n\n"
        if missing:
            msg += "Missing Keys:\n"

            for path in missing:
                msg += "\t"
                msg += join_path(path)
                msg += "\n"

            msg += "\n"

        if invalid_types:
            msg += "Invalid Types:\n"

            for item in invalid_types:
                msg += "\t"
                msg += join_path(item['path']) + ": "
                value = item['value']
                required_type = format_type(item['required_type'])
                actual_type = format_type(item['actual_type'])
                msg += "'{value}' should be of type '{required_type}' but was '{actual_type}'".format(
                    value=value, required_type=required_type, actual_type=actual_type)
                msg += "\n"

            msg += "\n"

        return msg

    def _validate(self, data):
        missing_paths = []
        invalid_types = []

        flat_schema = flatten_dict(self._meta_schema)
        for path, value_spec in flat_schema:
            value = get_path(data, path)
            if not value:
                if value_spec.get('required'):
                    missing_paths.append(path)
                continue

            actual_type = type(value)
            if 'enum' in value_spec:
                enum = value_spec['enum']
                if value not in value_spec['enum']:
                    invalid_types.append(
                        {"path": path,
                         "required_type": enum,
                         "value": value,
                         "actual_type": actual_type})
                continue

            if 'type' in value_spec:
                required_type = value_spec['type']
                if required_type == file:
                    required_type = str
                if actual_type != required_type:
                    invalid_types.append(
                        {"path": path,
                         "required_type": required_type,
                         "value": value,
                         "actual_type": actual_type})

        return missing_paths, invalid_types

    def _click_option_args(self, options, help_prefix="Metadata:"):

        all_options = set(self._flat_meta_schema.keys())
        options = set(options)
        bad_options = options - all_options
        if bad_options:
            raise RuntimeError("Unknown meta schema options: %s" %
                               " ".join(bad_options))

        for option in options:
            schema = self._flat_meta_schema[option]
            args = ["--%s" % option.replace("_", "-")]
            kwargs = {k: v for k, v in schema.iteritems() if k in [
                'required', 'default', 'type']}

            if 'description' in schema:
                kwargs["help"] = " ".join([help_prefix, schema["description"]])
            if 'enum' in schema:
                kwargs["type"] = click.Choice(schema['enum'])

            if schema.get("type") == file:
                exists = schema.get('exists', False)
                kwargs["type"] = click.Path(exists=exists)

            yield (args, kwargs)

    def get_schema(self, key):
        """ Get schema for a flat key
        """
        return self._flat_meta_schema.get(key)

    def get_default(self, key):
        """ Get default value for a flat key
        """
        schema = self.get_schema(key)
        if type(schema) == dict:
            return schema.get('default')

    def _create_meta(self, options={}, schema=None, path=[], meta={}):
        """ Generate nested meta dict.

            Arguments:
                options (dict): Additional flat options to be merge in
                                the meta data.
                schema (dict): Partial subschema (recursion argument)
                path (list): Current path (recursion argument)
                meta (dict): Current genereated meta data (recursion argument)

        """
        if not schema:
            schema = self._meta_schema

        meta = {}
        for key, value in schema.iteritems():
            if type(value) == dict and type(value.values()[0]) == dict:
                meta[key] = self._create_meta(
                    options, value, path=path + [key], meta=meta)
            else:
                option_key = "_".join(path + [key])
                if option_key in options:
                    meta[key] = options[option_key]
                else:
                    meta[key] = self.get_default(option_key)

        return meta


class ValidationError(Exception):
    pass
