PROJECT = faststats
cleantemp = rm -rf build; rm -f *.c
DIRS= core include
.PHONY : clean all build
all: clean build

build:  
	for d in $(DIRS); do (cd $$d; $(MAKE) build );done
	python setup.py build
	$(cleantemp)
clean: 
	for d in $(DIRS); do (cd $$d; $(MAKE) clean );done
	$(cleantmp)
	find . -name '*pyc' -exec rm -f {} \;
