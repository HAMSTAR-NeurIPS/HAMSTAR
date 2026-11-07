#!/bin/bash
# Reproduce all main-table results.
set -e

bash scripts/etth1.sh
bash scripts/etth2.sh
bash scripts/ettm1.sh
bash scripts/ettm2.sh
bash scripts/weather.sh
bash scripts/electricity.sh
bash scripts/solar.sh
bash scripts/czelan.sh
bash scripts/temp.sh
bash scripts/pm2_5.sh
