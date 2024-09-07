#!/bin/bash
export AIRFLOW_HOME=$(pwd)/airflow

airflow db init
airflow users create -e madhabpoulikwork@gmail.com -f madhab -l poulik -p admin -r Admin -u admin
nohup airflow scheduler &
airflow webserver