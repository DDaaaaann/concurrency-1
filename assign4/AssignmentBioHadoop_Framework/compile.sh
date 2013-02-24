#!/bin/bash
LIB=lib
SRC=src
CLASSPATH=$(echo "$LIB"/*.jar | tr ' ' ':')
PACKAGE_NAME=nl.uva
MAIN_CLASS=$PACKAGE_NAME.AssignmentMapreduce

javac -cp $CLASSPATH -d . $SRC/nl/uva/*.java


echo "Main-Class: $MAIN_CLASS" > manifest
JAR_CLASSPATH=$(echo "$LIB"/*.jar | tr ' ' ' ')
echo "Class-Path: $JAR_CLASSPATH" >> manifest

jar cvfm $MAIN_CLASS.jar manifest nl/uva/*.class lib


rm -r  nl
rm manifest