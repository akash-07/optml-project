#!/bin/bash
dataset=$1
r=$2
b=$3
l=$4
u=$5
fr=$6
runl=$7
runr=$8

LOGROOT=../logs/$(whoami)/fedavg/

for ((i=$runl; i<=$runr; i++))
	do	        
		python main.py -d $dataset -traindir ../leaf/data/${dataset}/data/train/ -testdir ../leaf/data/${dataset}/data/test/ -b $b -r $r -ee 2 -n 20 -lb $l -up $u -l ${LOGROOT}/${dataset}/${l}_${u}_lr${r}/r${i} -mwf ../model_weights/${dataset}/m${i}.h5 -sd $i -f $fr
	done

