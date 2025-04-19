import:py numpy;
import:py matplotlib;
import time;
import:py from jaclang.runtimelib.gins.smart_assert, smart_assert ;

with entry {
    can calculate_empirical_risk(X:numpy.ndarray, y:numpy.ndarray, theta:numpy.ndarray) -> float {
        n = X.shape[0];
        sum = 0;
        for i in range(0, n) {
            val = y[i] - numpy.dot(theta, X[i]);
            val **= 2;
            val /= 2;
            sum += val;
        }

        smart_assert(n != 0);
        sum /= n;

        return sum;
    }

    can calculate_RMS_Error(X:numpy.ndarray, y:numpy.ndarray, theta:numpy.ndarray) -> float {
        n = X.shape[0];
        d = X.shape[1];
        E_rms = 0;
        for i in range(0, n) {
            E_rms += (numpy.dot(theta, X[i]) - y[i]) ** 2;
        }
        E_rms /= n;
        E_rms = numpy.sqrt(E_rms);

        return E_rms;
    }

    can generate_polynomial_features(X:numpy.ndarray, M:int) -> numpy.ndarray {
        n = X.shape[0];
        Phi = numpy.zeros((n, M+1));
        for i in range(0, n) {
            for j in range(0, M + 1) {
                Phi[i][j] = X[i] ** j;
            }
        }

        return Phi;
    }

    can print_output(theta:numpy.ndarray, learning_rate:float, numIt:int, time_change:float) {
        return;
        print(f"Learning Rate \u03B7: {learning_rate}");
        print(f"Number of Iterations: {numIt}");
        for i in range(0, theta.shape[0]) {
            print(f"\u03B8_{i} = {theta[i]}");
        }
        print(f"Time Elapsed: {time_change}");
        print();
    }

    can ls_gradient_descent(X:numpy.ndarray, y:numpy.ndarray, learning_rate:float=0) -> numpy.ndarray {
        start = time.time();
        n = X.shape[0];
        d = X.shape[1];

        # Initialize Variables
        theta = numpy.zeros((d,));
        prev_loss = calculate_empirical_risk(X, y, theta) + 1;
        new_loss = calculate_empirical_risk(X, y, theta);

        numIt = 0;
        while (numIt < 1e6) and (abs(new_loss - prev_loss) > 1e-10) {
            # Calculate Gradient of Empirical Loss
            update = 0;
            for i in range(0, n) {
                update += (y[i] - numpy.dot(theta, X[i])) * X[i];
            }
            # Update theta
            update /= n;
            update *= learning_rate;
            theta += update;

            # Update stopping criteria
            prev_loss = new_loss;
            new_loss = calculate_empirical_risk(X, y, theta);
            numIt += 1;
        }

        end = time.time();

        # Print Output
        # print("Gradient Descent");
        print_output(theta, learning_rate, numIt, end - start);
        return theta;
    }

    can ls_stochastic_gradient_descent(X:numpy.ndarray, y:numpy.ndarray, learning_rate:float=0) -> numpy.ndarray {
        start = time.time();
        n = X.shape[0];
        d = X.shape[1];

        # Initialize Variables
        theta = numpy.zeros((d,));
        prev_loss = calculate_empirical_risk(X, y, theta) + 1;
        new_loss = calculate_empirical_risk(X, y, theta);

        numIt = 0;
        while (numIt < 1e6) and (abs(new_loss - prev_loss) > 1e-10) {
            # Calculate Gradient of Empirical Loss

            for i in range(0, n) {
                update = (y[i] - numpy.dot(theta, X[i])) * X[i];

                # Update theta
                update *= learning_rate;
                theta += update;
                numIt += 1;
            }
            # Update stopping criteria
            prev_loss = new_loss;
            new_loss = calculate_empirical_risk(X, y, theta);
        }
        end = time.time();

        # Print Output
        # print("Stochastic Gradient Descent");
        print_output(theta, learning_rate, numIt, end - start);

        return theta;
    }

    can closed_form_optimization(X:numpy.ndarray, y:numpy.ndarray, reg_param:float=0) -> numpy.ndarray {
        n = X.shape[0];
        d = X.shape[1];
        theta = numpy.zeros((d,));

        start = time.time();
        X_T = X.transpose();
        inv = numpy.matmul(X_T, X);
        inv += reg_param * numpy.identity(d);
        inv = numpy.linalg.inv(inv);
        temp = numpy.matmul(X_T, y);
        theta = numpy.matmul(inv, temp);

        end = time.time();

        # print("Closed Form Solution");
        # print(f"Regularization Parameter \u03BB = {reg_param}");
        # for i in range(0, theta.shape[0]) {
        #     print(f"\u03B8_{i} = {theta[i]}");
        # }
        # print(f"Time Elapsed: {end - start}");
        # print();

        return theta;
    }

    train_data = [
        [0.6755294828444033,0.6054908010772743],
        [0.033827894115051604,0.901713727909418],
        [-0.0513840229887791,0.924490729474708],
        [-0.072651497611752,0.836797477307684],
        [0.1423462723132326,0.1122895856168532],
        [0.2801281690878424,0.7404775817588839],
        [0.6520401465484031,0.41580094200540807],
        [0.836398699520691,0.5965109454480806],
        [0.5194478071553773,0.703871465558813],
        [0.1376010145959681,0.2519302178829483],
        [0.3423136677254472,0.2853564020665839],
        [0.33078259516683234,0.7578059336011658],
        [0.0519201109658643,0.8069236094590105],
        [0.042514140974268894,0.8801676397749953],
        [0.0178649580375613,0.7910048939943627],
        [0.3785882155791389,0.1067443064210857],
        [-0.0539132582155979,0.14573373513289511],
        [0.243761707490169,0.762960699487701],
        [0.1327927056707958,0.2936982161996094],
        [0.0865229852553948,0.2776060139421467],
    ];

    val_data = [
        [0.0758615810892394,0.9131455311697625],
        [0.32412388912616896,0.9817790261143028],
        [0.5815351563241093,0.5836868580111579],
        [-0.0888426458342539,0.25302051984997653],
        [-0.0913273424895036,0.8254519331384003],
        [0.6233452139803044,0.5306210396145495],
        [0.43832817850954736,0.6739730655635453],
        [0.0182433373473472,0.850297922913706],
        [0.7060528225986517,0.47095939629243067],
        [0.8850646556500316,0.5190600733073051],
        [0.31842362252982065,0.7592511175532508],
        [0.5918652234930168,0.6543059141233301],
        [0.3520259534361145,0.338075238744354],
        [0.0479374633676677,0.0900446669390177],
        [0.0450017343273448,0.8166451383585628],
        [0.6473511579137959,0.4993041322996459],
        [0.5553047378809265,0.3517664077336467],
        [0.8706742569243444,0.5601610561751518],
        [0.115715739472674,0.3325088286600567],
        [0.2936713698202222,0.3892315500462974],
    ];


    print("========== Part 1 ==========");

    X_train = numpy.array([]);
    y_train = numpy.array([entry[1] for entry in train_data]);

    bigX = generate_polynomial_features(X_train, 1);
    for i in range(-3, 0) {
        theta_GD = ls_gradient_descent(bigX, y_train, 10**i);
        theta_SGD = ls_stochastic_gradient_descent(bigX, y_train, 10**i);
    }
    theta_closed = closed_form_optimization(bigX, y_train);
    y_pred = [];
    for i in bigX {
        y_pred.append(numpy.dot(theta_closed, i));
    }

    print("Done!");

    print("=========== Part 2 ==========");

    X_validation = numpy.array([entry[0] for entry in val_data]);
    y_validation = numpy.array([entry[1] for entry in val_data]);

    errors_train = numpy.zeros((11,));
    errors_validation = numpy.zeros((11,));

    for i in range(0, 11) {
        X_tr = generate_polynomial_features(X_train, i);
        X_test = generate_polynomial_features(X_validation, i);
        theta = closed_form_optimization(X_tr, y_train);
        errors_train[i] = calculate_RMS_Error(X_tr, y_train, theta);
        errors_validation[i] = calculate_RMS_Error(X_test, y_validation, theta);
    }

    errors_train = numpy.zeros((10,));
    errors_validation = numpy.zeros((10,));
    L = numpy.append([0], 10.0 ** numpy.arange(-8, 1));

    X_tr = generate_polynomial_features(X_train, 10);
    X_test = generate_polynomial_features(X_validation, 10);
    for i in range(0, 10) {
        theta = closed_form_optimization(X_tr, y_train, L[i]);
        errors_train[i] = calculate_RMS_Error(X_tr, y_train, theta);
        errors_validation[i] = calculate_RMS_Error(X_test, y_validation, theta);
    }

    print("Done!");
}
