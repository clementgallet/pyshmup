from distutils.core import setup, Extension

draw = Extension('draw', sources = ['draw.c'],
libraries = ['GL']
	)

setup(name = 'draw',
	version = '0.1',
	description = 'This is also a demo package',
	ext_modules = [draw])
