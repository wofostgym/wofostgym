#!/bin/bash

for i in $(seq 1 15);
do
    python3 gen_data.py --npk.crop-rand --npk.scale .25 --agro-file "$1"_agro.yaml --npk.seed "$i" --year-low 1984 --year-high 2023 --policy-name BiWeekly_NW --save-folder kristen_data_new/"$1"/"$i"/ --data-file "$i"_good --file-type csv --npk.output-vars "[FIN, DVS, WSO, SM, NAVAIL, LAI, NAMOUNTRT, NDEMANDST, PDEMANDSO, RNUPTAKE, RKUPTAKERT, RPUPTAKELV, KDEMANDLV, PTRANSLOCATABLE, PAVAIL, PERCT, WRT, GRLV]" --npk.weather-vars "[IRRAD, TEMP, TMIN, TMAX, VAP,RAIN, WIND]" &
done

for i in $(seq 16 20);
do
    python3 gen_data.py --npk.crop-rand --npk.scale .5 --agro-file "$1"_agro.yaml --npk.seed "$i" --year-low 1984 --year-high 2023 --policy-name BiWeekly_NW --save-folder kristen_data_new/"$1"/"$i"/ --data-file "$i"_bad --file-type csv --npk.output-vars "[FIN, DVS, WSO, SM, NAVAIL, LAI, NAMOUNTRT, NDEMANDST, PDEMANDSO, RNUPTAKE, RKUPTAKERT, RPUPTAKELV, KDEMANDLV, PTRANSLOCATABLE, PAVAIL, PERCT, WRT, GRLV]" --npk.weather-vars "[IRRAD, TEMP, TMIN, TMAX, VAP,RAIN, WIND]" &
done