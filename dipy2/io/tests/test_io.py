"""Tests for overall io sub-package."""

from dipy2 import io


def test_imports():
    # Make sure io has not pulled in setup_module from dpy
    assert not hasattr(io, 'setup_module')
