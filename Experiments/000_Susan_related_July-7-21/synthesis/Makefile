# Given no targets, 'make' will default to building 'simv', the simulated version
# of the pipeline

# make          <- compile (and run) simv if needed

# As shortcuts, any of the following will build if necessary and then run the
# specified target

# make sim      <- runs simv (after compiling simv if needed)
# make dve      <- runs DVE interactively (after compiling it if needed)
#                                

# make clean    <- remove files created during compilations (but not synthesis)
# make nuke     <- remove all files created during compilation and synthesis
#
# synthesis command not included in this Makefile
#

################################################################################
## CONFIGURATION
################################################################################

#VCS = SW_VCS=2017.12-SP2-1 vcs -sverilog +vc -Mupdate -line -full64
VCS = SW_VCS=2017.12-SP2-1 vcs +v2k -sverilog +vc -Mupdate -line -full64
LIB = /afs/umich.edu/user/b/a/bakertim/Private/Synopsys/pdk_v1.3_v2010_12/NangateOpenCellLibrary_PDKv1_3_v2010_12/Front_End/Verilog/NangateOpenCellLibrary.v

# TODO: Update these depending on which file you want to synthesize. can also specify these params on command line rather than here.
export CLOCK_PERIOD=1000
export n=8
export q=8
export file_ID=FIR1
export subdir=IFIR_experiments1_Jul21
export grouped=1
export core=1

export v_filename=cemux_${file_ID}_n${n}_q${q}
export VER_DIR=verilog/${subdir}
export TCL_DIR=tcl
export DDC_DIR=ddc/${subdir}

export OUT_DIR=outs/${subdir}
export VG_DIR=vg/${subdir}


# Pick the TCL File. If c=core=1, then core_filter_synth.tcl is always used.
# If c=core=0 then g=grouped decides the TCL file
ifeq (${grouped},1)
	TCL_FILE = grouped_filter_synth.tcl
	export filenamer=cemux_${file_ID}_n${n}_q${q}_g${grouped}
endif
ifeq (${grouped},0)
	TCL_FILE = ungrouped_filter_synth.tcl
	export filenamer=cemux_${file_ID}_n${n}_q${q}_g${grouped}
endif
ifeq (${core},1)
	TCL_FILE = core_filter_synth.tcl
	export filenamer=cemux_${file_ID}_n${n}_q${q}_c${core}
endif

# SIMULATION CONFIG
SIMFILES = ${VER_DIR}/${filenamer}.v
SYNFILES = ${VG_DIR}/${filenamer}.vg
TESTBENCH = verilog/benches/blank.v

$(info $$filenamer is [${filenamer}])
$(info $$TCL_FILE is [${TCL_FILE}])

################################################################################
## RULES
################################################################################

# Default target:
all:	simv
	./simv | tee ${OUT_DIR}/program.out

.PHONY: all

# Simulation:

sim:	simv $(ASSEMBLED)
	./simv | tee ${OUT_DIR}/sim_program.out

simv:	$(HEADERS) $(SIMFILES) $(TESTBENCH)
	$(VCS) $^ -o simv

.PHONY: sim

# Debugging

#dve_simv:	$(SIMFILES) $(TESTBENCH)
#	$(VCS) +memcbk $(SIMFILES) -o -R -gui

#dve:	dve_simv $(ASSEMBLED)
#	./$<

dve:	$(SIMFILES) $(TESTBENCH)
	$(VCS) +memcbk $(TESTBENCH) $(SIMFILES) -o dve -R -gui

clean:
	rm -rvf simv *.daidir csrc vcs.key program.out \
	syn_simv syn_simv.daidir syn_program.out \
	*.vpd *.vcd *.dump ucli.key *.mr *.syn *.pvl *svf

nuke:	clean
	rm -rvf vg/${subdir}/* reports/${subdir}/* *.db chks/${subdir}/* *.log outs/${subdir}/* DVEfiles/ ddc/${subdir}/*

.PHONY: clean nuke dve

# Synthesis

syn_simv:	$(SYNFILES) $(TESTBENCH)
	$(VCS) $(TESTBENCH) $(SYNFILES) $(LIB) -o syn_simv

syn:	syn_simv
	./syn_simv | tee ${OUT_DIR}/syn_program.out

${VG_DIR}/${filenamer}.vg: ${VER_DIR}/${v_filename}.v ${TCL_DIR}/${TCL_FILE}
	dc_shell-t -f ${TCL_DIR}/${TCL_FILE} | tee ${OUT_DIR}/${filenamer}.out
