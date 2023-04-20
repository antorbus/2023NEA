.PHONY = all clean

PYS = go.py

all: $(PYS)

%.py: %.ipynb
	        jupyter nbconvert --RegexRemovePreprocessor.patterns="['^#noexport']" --to script $<

clean:
	        $(RM) $(PYS)
