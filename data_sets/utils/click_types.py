import click
import ast


def parse_list(value):
    """Parse list literal
    """

    as_list = ast.literal_eval(value)
    if type(as_list) is not list:
        raise ValueError("Value is not a list")
    return as_list


def eval_element(value, allowed_types=[int, float, str]):
    """Evaluate a single element.

    Try to parse as python literal or return as string.

    Args:
        value: Value to evaluate
        allowed_types: Types, that are allowed

    Return:
        Evaluated literal or value
    """

    try:
        evaluated = ast.literal_eval(value)
        if(type(evaluated) in allowed_types):
            return evaluated
    except (ValueError, SyntaxError):
        pass

    return value


class Collection(click.ParamType):

    name = 'collection'

    def convert(self, value, param, ctx):
        """Try to convert the input value to a list

        Examples:

            "a,b,c" => ['a', 'b', 'c']
            "1, 2, 3" => [1, 2, 3]
            "[1, 2, 3]" => [1, 2, 3]
            "foo" => ['foo']
            [1, 2, 3] => [1, 2, 3]

        Exceptions:
            ValueError

        """
        if isinstance(value, list):
            return value
        try:
            return parse_list(value)
        except (ValueError, SyntaxError):
            pass

        return [eval_element(elem.strip()) for elem in value.split(',')]
