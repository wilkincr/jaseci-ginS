with entry {
#   Variable values in this block:   v0 = 300 (seen 9×)  c_d = 0.001 (seen 8×)  runwayLength = 100 (seen 7×)  tolerance = 1e-06 (seen 6×)  timeStep = 1e-05 (seen 5×)  bmin = 0 (seen 4×)  bmax = 450.0 (seen 3×)  bmid = 225.0 (seen 2×)  round = 0 (seen 1×)
    v0:float = 300;
    c_d:float = 0.001;
    runwayLength:float = 100;

    tolerance:float = 0.000001;
    timeStep:float = 0.00001; 
    bmin:float = 0;
    bmax:float = v0 * v0 / (2.0 * runwayLength);
    bmid:float = (bmin + bmax) / 2.0;
    round:int = 0;

    while ((bmax - bmin) > tolerance and round < 10) {
        round += 1;
        #   Variable values in this block:   v0 = 300 (seen 11×)  c_d = 0.001 (seen 11×)  runwayLength = 100 (seen 11×)  tolerance = 1e-06 (seen 11×)  timeStep = 1e-05 (seen 11×)  bmin = 0 (seen 11×)  bmax = 450.0 (seen 11×)  bmid = 225.0 (seen 11×)  round = 1 (seen 10×)  round = 0 (seen 1×)  a_old = 0 (seen 5×)  a_old = -315.0 (seen 1×)  a_old = -225.000000009 (seen 1×)  a_old = -225.0 (seen 1×)  v_old = 300 (seen 4×)  v_old = 0.003 (seen 1×)  v_old = 3.0000000000000004e-08 (seen 1×)  v_old = 3.000000000000001e-13 (seen 1×)  x_old = 0 (seen 6×)  v_new = 0 (seen 2×)  v_new = 299.99685 (seen 1×)  v_new = 0.0007499999999100002 (seen 1×)  v_new = -0.0022499700000000004 (seen 1×)  x_new = 0 (seen 1×)  x_new = 0.003 (seen 1×)  x_new = 3.0000000000000004e-08 (seen 1×)  x_new = 3.000000000000001e-13 (seen 1×)
        bmid:float = (bmin + bmax) / 2.0;
        a_old:float = 0;
        v_old:float = v0;
        x_old:float = 0;
        v_new:float = 0;
        x_new:float = 0;

        while (v_old > 0 and x_old < runwayLength) {
            a_old:float = - bmid - c_d * v_old * v_old;
            #   Variable values in this block:   v0 = 300 (seen 16×)  c_d = 0.001 (seen 16×)  runwayLength = 100 (seen 16×)  tolerance = 1e-06 (seen 16×)  timeStep = 1e-05 (seen 16×)  bmin = 0 (seen 16×)  bmax = 450.0 (seen 16×)  bmid = 225.0 (seen 16×)  round = 1 (seen 16×)  a_old = -315.0 (seen 5×)  a_old = -225.000000009 (seen 5×)  a_old = -225.0 (seen 5×)  a_old = 0 (seen 1×)  v_old = 300 (seen 4×)  v_old = 0.003 (seen 4×)  v_old = 3.0000000000000004e-08 (seen 4×)  v_old = 3.000000000000001e-13 (seen 1×)  v_old = 299.99685 (seen 1×)  x_old = 0 (seen 16×)  v_new = 299.99685 (seen 5×)  v_new = 0.0007499999999100002 (seen 5×)  v_new = -0.0022499700000000004 (seen 4×)  v_new = 0 (seen 2×)  x_new = 0.003 (seen 5×)  x_new = 3.0000000000000004e-08 (seen 5×)  x_new = 0 (seen 3×)  x_new = 3.000000000000001e-13 (seen 3×)
            v_new:float = v_old + a_old * timeStep;
            x_new:float = x_old + v_old * timeStep;
            v_old:float = v_new;
            v_old:float = x_new;
        }
    
        if (x_old > runwayLength){
            bmin:float = bmid;
        }

        elif (x_old < runwayLength){
            bmax:float = bmid;
        }
    }

  print(f"b_optimal = {bmax}");
}