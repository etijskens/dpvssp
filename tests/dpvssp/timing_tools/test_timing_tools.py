# -*- coding: utf-8 -*-

"""Tests for (sub)module dpvssp.timing_tools."""

import sys
sys.path.insert(0,'.')

import dpvssp.timing_tools as timing_tools


def test_hello_default_arg():
    result = timing_tools.hello()
    assert result == "Hello world!"


def test_hello_me():
    result = timing_tools.hello('me')
    assert result == "Hello me!"


# ==============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (otherwise all tests are normally run with pytest)
# Make sure that you run this code with the project directory as CWD, and
# that the source directory is on the path
# ==============================================================================
if __name__ == "__main__":
    the_test_you_want_to_debug = test_hello_default_arg

    print("__main__ running", the_test_you_want_to_debug)
    the_test_you_want_to_debug()
    print('-*# finished #*-')

# eof