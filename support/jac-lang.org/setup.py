"""A Pygments lexer for the Jac programming language."""
from setuptools import setup

setup(
    name="jac-highlighter",
    version="0.0.1",
    description="A Pygments lexer for the Jac programming language.",
    py_modules=["jac_syntax_highlighter"],
    install_requires=["pygments"],
    entry_points="""
        [pygments.lexers]
        jaclexer = jac_syntax_highlighter:JacLexer
        """,
)
