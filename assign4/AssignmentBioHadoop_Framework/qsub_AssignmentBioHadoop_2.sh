#!/bin/bash 

HADOOP_HOME=/var/scratch/$USER/hadoop-1.0.4/
BASE_DIR=$HOME/workspace/Hadoop/AssignmentBioHadoop_2/
NUM_OF_REC=47550 
NUM_OF_SPLIT=$1
REC_PER_SPLIT=$(( $NUM_OF_REC / $NUM_OF_SPLIT )) 
echo $REC_PER_SPLIT
DATE=`date +%s`

export PATH=/usr/local/package/jdk1.6.0_17-linux-amd64/bin:${PATH}


time $HADOOP_HOME/bin/hadoop  jar $BASE_DIR/nl.uva.AssignmentMapreduce.jar /data/query.fa /data/fasta/ /data/BLOSUM100 10 /data/out-$DATE -recordsPerSplit $REC_PER_SPLIT
