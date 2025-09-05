# Energy-balance python codes


Set of python codes to find the optimzal size of battery for a solar+wind renewable generation with and without grid connectivity.

File names: -

Pb1.py:-Optimization code for 4 hour storage that runs on the power system with grid connectivity

Pb2.py:- Final power balance graphical output on battery power and energy rating

Pb3.py:-Optimization code for Long term storage without grid connectivity

Pb4.py:-Final power balance graphical output on battery power and energy rating


For the two optimization codes:-

(n_particles=2,max_iter=2)   #Change number of particles and number of iterations to lower run time

For one particle the run time is around 10 min, the code might run for 6 hours for more particles and iterations so take that into consideration.
 

data files:-
13-copy -> Orginal excell data file unscalled 2024
14 -copy -> Scaled 90% penetration renewable generation
15 - 100%renewable.xlsx ---> scaled 100% penetration renewable generation
16 - 130%renewable.xlsx ---> scaled 130% penetration renewable generation
scaled_renewables.xlsx ---> the output excel file from (pb3 finding scale.py) python code which scales the data based on requirement.
TEST DATA.xlsx  --> test data used to check charge cycle calculation or power balance graph.

im lazy to edit this , reach out to me on adarshautobot@gmail.com
