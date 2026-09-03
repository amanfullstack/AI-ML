#!/bin/bash 
kill $(lsof -t -i:8108) 
python app/manage.py runserver 0.0.0.0:8108
