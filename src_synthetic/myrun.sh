#!/bin/bash
dataset=$1
r=$2
b=$3
l=$4
u=$5
fr=$6
runl=$7
runr=$8

LOGROOT=/mnt/nfs/$(whoami)/optml/logs/fedavg

for ((i=$runl; i<=$runr; i++))
	do	        
		python main.py -d $dataset -traindir /mnt/nfs/dhasade/optml/data/${dataset}/train/ -testdir /mnt/nfs/dhasade/optml/data/${dataset}/test/ -b $b -r $r -ee 1 -n 5 -lb $l -up $u -l ${LOGROOT}/${dataset}/${l}_${u}_lr${r}/r${i} -sd $i -f $fr
	done

