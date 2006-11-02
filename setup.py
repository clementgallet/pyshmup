from distutils.core import setup, Extension

draw = Extension('draw', sources = ['draw.c'],
libraries = ['GL']
	)

setup(name = 'draw',
	version = '0.1',
	description = 'This is also a demo package',
	ext_modules = [draw])

coll = Extension('coll', sources = ['coll.c'],
	)

setup(name = 'coll',
	version = '0.1',
	description = 'This is also a demo package',
	ext_modules = [coll])

collml = Extension('collml', sources = ['collml.c'],
	)

setup(name = 'collml',
	version = '0.1',
	description = 'This is also a demo package',
	ext_modules = [collml])
