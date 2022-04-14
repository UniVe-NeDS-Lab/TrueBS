set xlabel 'gNB/km^2'
set datafile separator ' '
set term pdf font "Helvetica,13" size 4,3
set format x "%.0f"
set xrange [10:195]
set key inside top left

basedir='../results/results_tnsm/tnsm'

##Coverage
set key inside bottom right
do for [ratio in '0.5 2.0 4.0 100.0']{
    do for [ncov in '1 3']{
        set output basedir.'/coverage_'.ratio.'_'.ncov.'.pdf'
        set mytics 2
        set grid mytics
        set grid xtics ytics

        set ylabel 'coverage'
        set format y "%.1f"
        if(ncov==1){
            set yrange [0.5:1]
        }
        if(ncov==3){
            set yrange [0:.9]
        }

        plot basedir.'/urban_coverage_'.ratio.'_'.ncov.'.csv' using 1:2:6 with yerrorbars t 'r1 k=1' lc 'red',\
        '' using 1:2 with lines lc 'red' notitle,\
        '' using 1:3:7 with yerrorbars t 'r1 k=3' lc 'blue',\
        '' using 1:3 with lines lc 'blue' notitle,\
        '' using 1:4:8 with yerrorbars t 'rlc k=3' lc 'green',\
        '' using 1:4 with lines lc 'green' notitle,\
        '' using 1:5:9 with yerrorbars t 'fi k=3' lc 'pink',\
        '' using 1:5 with lines lc 'pink' notitle,\

    }
}
unset mytics
unset grid
set yrange [0:2.5]
do for [ratio in '4.0 100.0']{
    ## Cost
    set output basedir.'/cost_'.ratio.'.pdf'
    set ylabel '£ (Millions)'
    plot basedir.'/urban_cost_'.ratio.'.csv' using 1:2:6 with yerrorbars t 'r1 k=1' lc 'red',\
        '' using 1:2 with lines lc 'red' notitle,\
        '' using 1:3:7 with yerrorbars t 'r1 k=3' lc 'blue',\
        '' using 1:3 with lines lc 'blue' notitle,\
        '' using 1:4:8 with yerrorbars t 'rlc k=3' lc 'green',\
        '' using 1:4 with lines lc 'green' notitle,\
        '' using 1:5:9 with yerrorbars t 'fi k=3' lc 'pink',\
        '' using 1:5 with lines lc 'pink' notitle,\
    '' using 1:10 with lines title 'UB' lc 'black' dt 2,\
    '' using 1:11 with lines title 'LB' lc 'black' dt 4
}
unset yrange
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



# ##Cost on coverage
# do for [ncov in '1 2 3']{
#     set output basedir.'/costcoverage_'.ncov.'.pdf'

#     set ylabel '£ / m^2'
#     unset format y
#     #set logscale y 10
#     plot basedir.'/urban_costcoverage_'.ncov.'.csv' using 1:3 with lines t '5%' lc 'blue',\
#     '' using 1:4 with lines t '10%' lc 'green',\
#     '' using 1:5 with lines t '100%' lc 'pink',\

# }


#Plot scatter
set key inside top left
unset xrange
unset format x
unset yrange
set xrange[0.2:1]
set ylabel 'cost (Mln £)'
set xlabel 'coverage'
set bars small
set grid
set grid mxtics
#set xtics 0,.05,1
set mxtics 2
set output basedir.'/scatter_mean_1_r1.pdf'
plot basedir.'/scatter_mean_1r1.csv' using 1:2:3:4 with xyerrorbars title '0.5%' lc 'red' pt 7 ps 0.5,\
     '' using 1:2 with lines notitle lc 'red',\
     '' using 5:6:7:8 with xyerrorbars title '2%' lc 'blue' pt 7 ps 0.5,\
     '' using 5:6 with lines notitle lc 'blue',\
     '' using 9:10:11:12 with xyerrorbars title '4%' lc 'green' pt 7 ps 0.5,\
     '' using 9:10 with lines notitle lc 'green',\
     '' using 13:14:15:16 with xyerrorbars title '100%' lc 'pink' pt 7 ps 0.5,\
     '' using 13:14 with lines notitle lc 'pink',\


#Plot scatter
unset xrange
unset format x
unset yrange
set xrange[0.2:1]
set ylabel 'cost (Mln £)'
set xlabel 'coverage'
set bars small
set grid
set grid mxtics
#set xtics 0,.05,1
set mxtics 2
set output basedir.'/scatter_mean_3_rlc.pdf'
plot basedir.'/scatter_mean_3rlc.csv' using 1:2:3:4 with xyerrorbars title '0.5%' lc 'red' pt 7 ps 0.5,\
     '' using 1:2 with lines notitle lc 'red',\
     '' using 5:6:7:8 with xyerrorbars title '2%' lc 'blue' pt 7 ps 0.5,\
     '' using 5:6 with lines notitle lc 'blue',\
     '' using 9:10:11:12 with xyerrorbars title '4%' lc 'green' pt 7 ps 0.5,\
     '' using 9:10 with lines notitle lc 'green',\
     '' using 13:14:15:16 with xyerrorbars title '100%' lc 'pink' pt 7 ps 0.5,\
     '' using 13:14 with lines notitle lc 'pink',\

set format x "%.0f"
set format y "%.2f"
unset xtics
unset xtics
unset yrange
unset xrange
set key inside top right
set ylabel ''
set xlabel ''
set xtics 0,2,24
set output basedir.'/threestep_4_75_distribution.pdf'
plot basedir.'/threestep_4_75_distribution.csv' using 1:2 with lines  title 'r1 k=1', \
     '' using 1:3 with lines title 'r1 k=3' ,\
     '' using 1:4 with lines title 'rlc k=3',\
     '' using 1:5 with lines title 'fi k=3',\
