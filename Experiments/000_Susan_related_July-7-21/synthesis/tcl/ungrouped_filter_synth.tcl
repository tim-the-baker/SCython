#/***********************************************************/
#/*   FILE        : synth.tcl                          */
#/*   Description : Default Synopsys Design Compiler Script */
#/*   Usage       : dc_shell -tcl_mode -f default.tcl       */
#/*   You'll need to minimally set design_name & read files */
#/***********************************************************/

#/***********************************************************/
#/* The following five lines must be updated for every      */
#/* new design                                              */
#/***********************************************************/

set search_path [ list "./" "/afs/umich.edu/user/b/a/bakertim/Private/Synopsys/"]

set filenamer $env(filenamer)
set v_filename $env(v_filename)
set VER_DIR $env(VER_DIR)
set VG_DIR $env(VG_DIR)
set ddc_dir $env(DDC_DIR)
set subdir $env(subdir)

analyze -f sverilog [list "${VER_DIR}/${v_filename}.v"]
elaborate filter

set design_name filter
set clock_name clock
set reset_name reset
set CLK_PERIOD $env(CLOCK_PERIOD)
set rep_dir reports/${subdir}/
set chk_dir chks/${subdir}/




#/***********************************************************/
#/* The rest of this file may be left alone for most small  */
#/* to moderate sized designs.  You may need to alter it    */
#/* when synthesizing your final project.                   */
#/***********************************************************/
set SYN_DIR ./
set target_library "NangateOpenCellLibrary.db"
set link_library [concat  "*" $target_library]

#/***********************************************************/
#/* Set some flags for optimisation */

set compile_top_all_paths "true"
set auto_wire_load_selection "true"


#/***********************************************************/
#/*  Clk Periods/uncertainty/transition                     */

set CLK_TRANSITION 0.1
set CLK_UNCERTAINTY 0.1
set CLK_LATENCY 0.1

#/* Input/output Delay values */
set AVG_INPUT_DELAY 0.1
set AVG_OUTPUT_DELAY 0.1

#/* Critical Range (ns) This tells DC to try to not only improve
#   the delay of the critical path, but also any path within CRIT_RANGE
#   of the critical path.*/
set CRIT_RANGE 1.0

#/***********************************************************/
#/* Design Constrains: Not all used                         */
set MAX_TRANSITION 1.0
set FAST_TRANSITION 0.1
set MAX_FANOUT 32
set MID_FANOUT 8
set LOW_FANOUT 1
set HIGH_DRIVE 0
set HIGH_LOAD 1.0
set AVG_LOAD 0.1
set AVG_FANOUT_LOAD 10

#/***********************************************************/
set DRIVING_CELL DFF_X1

#/* DONT_USE_LIST = {   } */

#/*************operation cons**************/
set LOGICLIB NangateOpenCellLibrary
#/*****************************/

#/* Sourcing the file that sets the Search path and the libraries(target,link) */

set sys_clk $clock_name

set netlist_file [format "%s%s"  [format "%s%s/%s"  $SYN_DIR $VG_DIR $filenamer] ".vg"]
set ddc_file [format "%s%s"  [format "%s%s/%s"  $SYN_DIR $ddc_dir $filenamer] ".ddc"]
set rep_file [format "%s%s"  [format "%s%s%s"  $SYN_DIR $rep_dir $filenamer] ".rep"]
set dc_shell_status [ set chk_file [format "%s%s"  [format "%s%s%s"  $SYN_DIR $chk_dir $filenamer] ".chk"] ]

#/* if we didnt find errors at this point, run */
if {  $dc_shell_status != [list] } {
  current_design $design_name
  link
  #set_wire_load_model -name $WIRE_LOAD -lib $LOGICLIB $design_name
  set_wire_load_mode top
  set_fix_multiple_port_nets -outputs -buffer_constants
  create_clock -period $CLK_PERIOD -name $sys_clk [find port $sys_clk]
  set_clock_uncertainty $CLK_UNCERTAINTY $sys_clk
  set_fix_hold $sys_clk
  group_path -from [all_inputs] -name input_grp
  group_path -to [all_outputs] -name output_grp
  set_driving_cell  -lib_cell $DRIVING_CELL [all_inputs]
  remove_driving_cell [find port $sys_clk]
  #set_fanout_load $AVG_FANOUT_LOAD [all_outputs]
  #set_load $AVG_LOAD [all_outputs]
  set_input_delay $AVG_INPUT_DELAY -clock $sys_clk [all_inputs]
  #remove_input_delay -clock $sys_clk [find port $sys_clk]
  set_output_delay $AVG_OUTPUT_DELAY -clock $sys_clk [all_outputs]
  set_dont_touch $reset_name
  set_resistance 0 $reset_name
  set_drive 0 $reset_name
  set_critical_range $CRIT_RANGE [current_design]
  #set_max_delay $CLK_PERIOD [all_outputs]
  #set_max_area -ignore_tns 0
  set MAX_FANOUT $MAX_FANOUT
  #set MAX_TRANSITION $MAX_TRANSITION
  uniquify
  ungroup -all -flatten
  redirect $chk_file { check_design }
  #compile -map_effort high -area_effort high
  compile_ultra
  write -hier -format verilog -output $netlist_file $design_name
  write -hier -format ddc -output $ddc_file $design_name
  redirect $rep_file { report_design -nosplit }
  redirect -append $rep_file { report_area }
  redirect -append $rep_file { report_timing -max_paths 2 -input_pins -nets -transition_time -nosplit }
  redirect -append $rep_file { report_constraint -max_delay -verbose -nosplit }
  redirect -append $rep_file { report_power -nosplit -hierarchy }
  remove_design -all
  read_file -format verilog $netlist_file
  current_design $design_name
  redirect -append $rep_file { report_reference -nosplit }
  quit
} else {
   quit
}


