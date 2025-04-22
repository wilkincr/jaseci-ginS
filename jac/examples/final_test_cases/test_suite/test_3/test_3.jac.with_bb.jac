with entry { # BB: 0 Execution frequency: 1 Total execution time: 0.003 ms
    v0:float = 300; # BB: 0
    c_d:float = 0.001; # BB: 0
    runwayLength:float = 100; # BB: 0

    tolerance:float = 0.000001; # BB: 0
    timeStep:float = 0.00001;  # BB: 0
    bmin:float = 0; # BB: 0
    bmax:float = v0 * v0 / (2.0 * runwayLength); # BB: 0
    bmid:float = (bmin + bmax) / 2.0; # BB: 0
    round:int = 0; # BB: 0

    while ((bmax - bmin) > tolerance and round < 10) { # BB: 0
        round += 1; # BB: 2 Execution frequency: 1 Total execution time: 0.004 ms
        bmid:float = (bmin + bmax) / 2.0; # BB: 2
        a_old:float = 0; # BB: 2
        v_old:float = v0; # BB: 2
        x_old:float = 0; # BB: 2
        v_new:float = 0; # BB: 2
        x_new:float = 0; # BB: 2

        while (v_old > 0 and x_old < runwayLength) { # BB: 2
            a_old:float = - bmid - c_d * v_old * v_old; # BB: 4 Execution frequency: 1 Total execution time: 0.002 ms
            v_new:float = v_old + a_old * timeStep; # BB: 4
            x_new:float = x_old + v_old * timeStep; # BB: 4
            v_old:float = v_new; # BB: 4
            v_old:float = x_new; # BB: 4
        }
    
        if (x_old > runwayLength){ # BB: 7 Execution frequency: 0 Total execution time: 0.000 ms
            bmin:float = bmid; # BB: 8 Execution frequency: 0 Total execution time: 0.000 ms
        }

        elif (x_old < runwayLength){ # BB: 9 Execution frequency: 0 Total execution time: 0.000 ms
            bmax:float = bmid; # BB: 10 Execution frequency: 0 Total execution time: 0.000 ms
        }
    }

  print(f"b_optimal = {bmax}"); # BB: 14 Execution frequency: 0 Total execution time: 0.000 ms
}