# Makefile for facelock program

# Project tree variables
OUTDIR=out
OBJDIR=$(OUTDIR)/obj
SRCDIR=src

# Compiler variables
CXX=g++
INCLUDE=-I/usr/include/opencv4
CXXFLAGS=
LDFLAGS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_videoio

PROG=facelock
ELFS=$(addprefix $(OUTDIR)/, $(PROG))
OBJS=$(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(wildcard $(SRCDIR)/*.cpp))

all: $(ELFS)

$(ELFS): $(OBJS)
	@echo "Building binary files"
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(SRCDIR)/%.hpp | $(OBJDIR)
	@echo "Building object files"
	$(CXX) $(INCLUDE) $(CXXFLAGS) -c -g -o $@ $<

$(SRCDIR)/%.hpp: ;

$(OBJDIR):
	mkdir -p $(OBJDIR)

.PHONY: clean

clean:
	@rm -rfv $(OUTDIR)
