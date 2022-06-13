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

LOGROOT=/mnt/nfs/$(whoami)/optml/logs/fedprox

for ((i=$runl; i<=$runr; i++))
	do	        
		python main.py -d $dataset -traindir /mnt/nfs/dhasade/optml/data/${dataset}/train/ -testdir /mnt/nfs/dhasade/optml/data/${dataset}/test/ -b $b -r $r -ee 1 -n 2 -lb $l -up $u -l ${LOGROOT}/${dataset}/${l}_${u}_lr${r}/r${i} -sd $i -f $fr -sm True -mu $mu #-mwf ../model_weights/${dataset}/m${i}.h5 
	done

