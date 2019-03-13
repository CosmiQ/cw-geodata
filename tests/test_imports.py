import sys


class TestImports(object):
    def test_imports(self):
        from cw_geodata.geo_utils import core
        from cw_geodata import label, mask, image, utils
