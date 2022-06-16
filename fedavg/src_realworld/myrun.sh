#!/bin/bash
dataset=$1
r=$2
b=$3
l=$4
u=$5
fr=$6
runl=$7
runr=$8
type=$9

LOGROOT=../../logs/fedavg/realworld

for ((i=$runl; i<=$runr; i++))
	do	        
		python main.py -d $dataset -traindir ../../leaf/data/${dataset}/data/train_${type}/ -testdir ../../leaf/data/${dataset}/data/test_${type}/ -b $b -r $r -ee 3 -n 20 -lb $l -up $u -l ${LOGROOT}/${dataset}/${type}/${l}_${u}_lr${r}/r${i} -sd $i -f $fr -mwf ../../model_weights/${dataset}/m${i}.h5 
	done

ß