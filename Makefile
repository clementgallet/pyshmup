.PHONY: play all clean

play:	all
	python main.py

all: constants.h tags
	python setup.py build_ext --inplace

constants.h: constants.py
	@echo Generating constants.h ...
	@echo "#ifndef __CONSTANTS_PY__\n#define __CONSTANTS_PY__\n\n" > constants.h
	@echo "/* This file has been generated automatically." >> constants.h
	@echo " * Do not edit it directly; changes should be made to constants.py\n */\n" >> constants.h
	@sed -e '/END_OF_PORTABLE_CONSTANTS/ Q' -e 's/^import .*//' -e 's/##.*$$//' -e 's!#\(.*\)$$!/* \1 */!' -e 's/^\(..*\)=\(..*\)$$/\#define \1 \2/' constants.py | uniq >> constants.h
	@echo '\n\n#endif\n' >> constants.h

tags:	*.py
	-ctags *.py

clean:
	$(RM) *.pyc *.pyo constants.h *.so
	$(RM) -r build
