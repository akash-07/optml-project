#!/bin/bash
dataset=$1
r=$2 # learning rate
b=$3 # batch size --> synthetic: 100
l=$4 # lower bound on steps
u=$5 # upper bound on steps
fr=$6 # fixed rounds : 50
runl=$7 # how many times do I want to run ? --> 1
runr=$8 # how many times do I want to run ? --> 1

LOGROOT=../../logs/fedavg/artificial

for ((i=$runl; i<=$runr; i++))
	do	        
		python main.py -d $dataset -traindir ../../synthetic_data/train -testdir ../../synthetic_data/test -b $b -r $r -ee 1 -n 2 -lb $l -up $u -l ${LOGROOT}/${dataset}/${l}_${u}_lr${r}/r${i} -sd $i -f $fr -sm True
	done

