"""Script that creates the styling utils."""

import seaborn as sns


def make_palette() -> dict:
    """
    Get the color palette used for styling.

    Returns
    -------
    dict
        A dictionary containing the styling parameters:
        - 'palette': The color palette used for plotting.
        - 'lines': The color of the lines in the plots.
        - 'text': The color of the text in the plots.
        - 'text_darker': A darker shade of the text color.
        - 'text_lighter': A lighter shade of the text color.
        - 'background': The background color of the plots.
    """
    green = '#66b392'
    orange = '#f4a08f'
    purple = '#c696de'
    blue = '#739fd8'
    palette = [green, orange, purple, blue]

    text_darker = '#282828'
    text = '#918c88'
    text_lighter = '#b2afac'
    lines = '#dad9d7'
    background = '#faf7f4'
    return {
        'palette': palette,
        'lines': lines,
        'text': text,
        'text_darker': text_darker,
        'text_lighter': text_lighter,
        'background': background,
    }


def apply_styling(colors: dict = None):
    """
    Apply styling to the pandas and seaborn libraries.

    Parameters
    ----------
    colors : dict, optional
        A dictionary containing the colors to use for styling. If not provided,
        the default color palette will be used. The dictionary should contain
        the following keys:
        - 'palette': The color palette used for plotting.
        - 'lines': The color of the lines in the plots.
        - 'text': The color of the text in the plots.
        - 'text_darker': A darker shade of the text color.
        - 'text_lighter': A lighter shade of the text color.
        - 'background': The background color of the plots.
    """
    colors = colors or make_palette()

    params = {
        'figure.figsize': (6, 4),
        'figure.dpi': 200,
        'savefig.pad_inches': 0.3,
        'figure.facecolor': colors.get('background'),
        'axes.facecolor': colors.get('background'),
        'axes.edgecolor': colors.get('lines'),
        'axes.labelcolor': colors.get('text'),
        'axes.titlecolor': colors.get('text_darker'),
        'text.color': colors.get('text'),
        'xtick.color': colors.get('text_lighter'),
        'ytick.color': colors.get('text_lighter'),
        'font.size': 9,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.linewidth': 0.6,
        'axes.grid': False,
    }
    sns.set_theme(
        context='notebook', style='white', palette=colors.get('palette'), rc=params
    )
