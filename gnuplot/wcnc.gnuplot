set xlabel 'gNB/km^2'
set datafile separator ' '
set term pdf font "Helvetica,13" size 4,3
set format x "%.0f"
set xrange [0:130]
set key inside top left


##Coverage
do for [ncov in '1 2 3']{
    set output '~/TNSM2022/results_final/wcnc/coverage_'.ncov.'.pdf'

    set ylabel 'coverage'
    set format y "%.1f"
    set yrange [0.5:1]

    plot '~/TNSM2022/results_final/wcnc/urban_coverage_'.ncov.'.csv' using 1:2:6 with yerrorbars t '1%' lc 'red',\
    '' using 1:2 with lines lc 'red' notitle,\
    '' using 1:3:7 with yerrorbars t '5%' lc 'blue',\
    '' using 1:3 with lines lc 'blue' notitle,\
    '' using 1:4:8 with yerrorbars t '10%' lc 'green',\
    '' using 1:4 with lines lc 'green' notitle,\
    '' using 1:5:9 with yerrorbars t '100%' lc 'pink',\
    '' using 1:5 with lines lc 'pink' notitle,\

}



## Cost
set output '~/TNSM2022/results_final/wcnc/cost.pdf'
set ylabel '£ (Millions)'
unset yrange
plot '~/TNSM2022/results_final/wcnc/urban_cost.csv' using 1:2:6 with yerrorbars t '1%' lc 'red',\
    '' using 1:2 with lines lc 'red' notitle,\
    '' using 1:3:7 with yerrorbars t '5%' lc 'blue',\
    '' using 1:3 with lines lc 'blue' notitle,\
    '' using 1:4:8 with yerrorbars t '10%' lc 'green',\
    '' using 1:4 with lines lc 'green' notitle,\
    '' using 1:5:9 with yerrorbars t '100%' lc 'pink',\
    '' using 1:5 with lines lc 'pink' notitle,\
 '' using 1:10 with lines title 'UB' lc 'black' dt 2,\
 '' using 1:11 with lines title 'LB' lc 'black' dt 4


# ##Building ratio

# set output '~/TNSM2022/results_develop/buildrat.pdf'

# set ylabel '|gNB| / |B|'
# set format y "%.2f"

# plot '~/TNSM2022/results_develop/urban_buildrat.csv' using 1:2:6 with yerrorbars t '1%' lc 'red',\
#     '' using 1:2 with lines lc 'red' notitle,\
#     '' using 1:3:7 with yerrorbars t '5%' lc 'blue',\
#     '' using 1:3 with lines lc 'blue' notitle,\
#     '' using 1:4:8 with yerrorbars t '10%' lc 'green',\
#     '' using 1:4 with lines lc 'green' notitle,\
#     '' using 1:5:9 with yerrorbars t '100%' lc 'pink',\
#     '' using 1:5 with lines lc 'pink' notitle,\



##Cost on coverage
do for [ncov in '1 2 3']{
    set output '~/TNSM2022/results_final/wcnc/costcoverage_'.ncov.'.pdf'

    set ylabel '£ / m^2'
    unset format y
    #set logscale y 10
    plot '~/TNSM2022/results_final/wcnc/urban_costcoverage_'.ncov.'.csv' using 1:3 with lines t '5%' lc 'blue',\
    '' using 1:4 with lines t '10%' lc 'green',\
    '' using 1:5 with lines t '100%' lc 'pink',\

}


#Plot scatter
unset xrange
unset format x
unset yrange
set xrange[0.5:1]
set ylabel 'cost (Mln £)'
set xlabel 'coverage'
set bars small
set grid
set grid mxtics
#set xtics 0,.05,1
set mxtics 2
set output '~/TNSM2022/results_final/wcnc/scatter_mean.pdf'
plot '~/TNSM2022/results_final/wcnc/scatter_mean.csv' using 1:2:3:4 with xyerrorbars title '1%' lc 'red' pt 7 ps 0.5,\
     '' using 1:2 with lines notitle lc 'red',\
     '' using 5:6:7:8 with xyerrorbars title '5%' lc 'blue' pt 7 ps 0.5,\
     '' using 5:6 with lines notitle lc 'blue',\
     '' using 9:10:11:12 with xyerrorbars title '10%' lc 'green' pt 7 ps 0.5,\
     '' using 9:10 with lines notitle lc 'green',\
     '' using 13:14:15:16 with xyerrorbars title '100%' lc 'pink' pt 7 ps 0.5,\
     '' using 13:14 with lines notitle lc 'pink',\