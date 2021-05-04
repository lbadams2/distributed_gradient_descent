#!/bin/bash

while :
do
  #docker stats csc724_project_grad_calc_1_1 | awk '{print $4}' >> gc8080_runtime_metrics/memory.txt
  docker stats csc724_project_grad_calc_1_1 --no-stream | awk 'FNR == 2 {print $4}' >> gc8080_runtime_metrics/memory.txt
  docker stats csc724_project_optimizer_1 --no-stream | awk 'FNR == 2 {print $4}' >> opt_runtime_metrics/memory.txt
  #docker stats csc724_project_grad_calc_1_1 --no-stream | awk 'FNR == 2 {print $3}' >> gc8080_runtime_metrics/cpu.txt
  #docker stats csc724_project_grad_calc_1_1 | cut -f4
  sleep 2
done
