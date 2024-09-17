"""This module contains utility functions for logging."""

PIPE, SPACE = '|', ' '
TAB = SPACE * 4
INDENT_MAP = {
    0: '',
    1: PIPE + SPACE,
}
for i, lvl in enumerate(range(2, 10), 1):
    INDENT_MAP[lvl] = PIPE + (TAB * i) + SPACE


def bprint(*args, level: int = 0, prefix: str = '', **kwargs) -> None:
    """
    A better print function that prints a message with a
    specified indentation level.

    Parameters
    ----------
    *args : tuple
        The message parts to be printed.
    level : int, optional
        The indentation level of the message. The default is 0.
    prefix : str, optional
        The user-defined prefix to be printed before the message. The default is ''.
    **kwargs : dict
        Additional keyword arguments to be passed to the print function.

    Returns
    -------
    None
        This function does not return anything.

    Examples
    --------
    >>> bprint("Hello, world!", level=1)
    | Hello, world!
    """
    indent = INDENT_MAP[level]
    if level == 0:
        indent_0 = indent + prefix
    else:
        indent_0 = indent if not prefix else indent[: -len(prefix)] + prefix

    sep = kwargs.get('sep', ' ')
    text = sep.join(map(str, args))
    for line, subtext in enumerate(text.splitlines()):
        if line == 0:
            print(indent_0, subtext, sep='', **kwargs)
        else:
            print(indent, subtext, sep='', **kwargs)


if __name__ == '__main__':
    bprint('Hola, \nmundo !', level=3, prefix='¡ ')  # Debug
