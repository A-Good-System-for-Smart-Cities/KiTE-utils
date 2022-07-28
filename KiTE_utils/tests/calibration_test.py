# import pytest
import logging

logging.basicConfig(level=logging.WARNING)


# est simulation setting -- create oracle class + randomly mk some rando classifier + add some bias -- see if can reject H0 (how big should b be to reject H0)
def test_basic():
    assert 3 == 1 + 2
