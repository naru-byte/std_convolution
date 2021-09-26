#!/bin/bash
dir=${1:-"./data/datasets/HMDB51"}
wget -o $dir/hmdb51_org.rar http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
rm $dir/hmdb51_org.rar
unrar e $dir/hmdb51_org.rar
ls $dir |grep rar|xargs -I {} unrar x $dir/{}
ls $dir |grep rar|xargs -I {} rm $dir/{}