#!/bin/bash
dataset=$1
r=$2 # learning rate
b=$3 # batch size --> synthetic: 100
l=$4 # lower bound on steps
u=$5 # upper bound on steps
fr=$6 # fixed rounds : 50
runl=$7 # how many times do I want to run ? --> 1
runr=$8 # how many times do I want to run ? --> 1

LOGROOT=/Users/renard/Documents/etudes/EPFLMA4/OML/project/optml-project/logroot

for ((i=$runl; i<=$runr; i++))
	do	        
		/Users/renard/miniconda3/envs/TFF/bin/python main.py -d $dataset -traindir /Users/renard/Documents/etudes/EPFLMA4/OML/project/optml-project/synthetic_data/train/ -testdir /Users/renard/Documents/etudes/EPFLMA4/OML/project/optml-project/synthetic_data/test/ -b $b -r $r -ee 1 -n 2 -lb $l -up $u -l ${LOGROOT}/${dataset}/${l}_${u}_lr${r}/r${i} -sd $i -f $fr -sm True #-mwf ../model_weights/${dataset}/m${i}.h5 
	done

