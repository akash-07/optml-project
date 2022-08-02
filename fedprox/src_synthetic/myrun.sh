#!/bin/bash
dataset=$1
r=$2
b=$3
l=$4
u=$5
mu=$6
fr=$7
runl=$8
runr=$9

LOGROOT=../../logs/fedprox/artificial

for ((i=$runl; i<=$runr; i++))
	do	        
		python main.py -d $dataset -traindir ../../synthetic_data/train -testdir ../../synthetic_data/test -b $b -r $r -ee 1 -n 2 -lb $l -up $u -l ${LOGROOT}/${dataset}/${l}_${u}_lr${r}/r${i} -sd $i -f $fr -sm True -mu $mu
	done

