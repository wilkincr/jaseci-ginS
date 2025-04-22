with entry {
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
        bmid:float = (bmin + bmax) / 2.0;
        a_old:float = 0;
        v_old:float = v0;
        x_old:float = 0;
        v_new:float = 0;
        x_new:float = 0;

        while (v_old > 0 and x_old < runwayLength) {
            a_old:float = - bmid - c_d * v_old * v_old;
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