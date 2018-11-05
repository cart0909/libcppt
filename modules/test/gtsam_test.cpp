#include <gtest/gtest.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/base/Vector.h>
#include <gtsam/inference/Symbol.h>
using namespace gtsam;
using namespace std;

TEST(gtsam, ex0) {
    // Create an empty nonlinear factor graph
    NonlinearFactorGraph graph;

    // Add a Gaussian prior on pose x_1
    Pose2 priorMean(0.0, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr priorNoise =
    noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
    graph.add(PriorFactor<Pose2>(1, priorMean, priorNoise));

    // Add two odometry factors
    Pose2 odometry(2.0, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr odometryNoise =
    noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
    graph.add(BetweenFactor<Pose2>(1, 2, odometry, odometryNoise));
    graph.add(BetweenFactor<Pose2>(2, 3, odometry, odometryNoise));

    // create (deliberatly inaccurate) initial estimate
    Values initial;
    initial.insert(1, Pose2(0.5, 0.0, 0.2));
    initial.insert(2, Pose2(2.3, 0.1, -0.2));
    initial.insert(3, Pose2(4.1, 0.1, 0.1));

    // optimize using Levenberg-Marquardt optimization
    Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();

    // Query the marginals
    Marginals marginals(graph, result);

    for(int i = 1; i < 4; ++i)
        std::cout << "x" << i << "covariance:\n" << marginals.marginalCovariance(i) << std::endl;
}

class GPSPose2Factor : public NoiseModelFactor1<Pose2> {
public:
    GPSPose2Factor(const SharedNoiseModel& model, Key j, const Point2& xy)
        : NoiseModelFactor1<Pose2>(model, j), mx_(xy.x()), my_(xy.y()) {}

    Vector evaluateError(const Pose2& q, boost::optional<Matrix&> H = boost::none) const {
        if(H) {
            *H = (Matrix23() << 1.0, 0.0, 0.0,
                                0.0, 1.0, 0.0).finished();
        }

        return Vector2(q.x() - mx_, q.y() - my_);
    }

    double mx_, my_;
};

TEST(gtsam, ex1) {
    // Create a factor graph container
    NonlinearFactorGraph graph;

    // odometry measurement noise model (covariance matrix)
    noiseModel::Diagonal::shared_ptr odomModel =
            noiseModel::Diagonal::Sigmas(Vector3(0.5, 0.5, 0.1));

    // Add odometry factors
    // Create odometry (Between) factors between consecutive poses
    // robot makes 90 deg right turns at x3 - x5
    graph.add(BetweenFactor<Pose2>(Symbol('x', 1), Symbol('x', 2), Pose2(5, 0, 0), odomModel));
    graph.add(BetweenFactor<Pose2>(Symbol('x', 2), Symbol('x', 3), Pose2(5, 0, 0), odomModel));

    // 2D 'GPS' measurement noise model, 2-dim
    noiseModel::Diagonal::shared_ptr gpsModel = noiseModel::Diagonal::Sigmas(Vector2(1.0, 1.0));

    // Add the GPS factors
    // note that there is NO prior factor needed at first pose, since GPS provides
    // the global positions (and rotations given more than 1 GPS measurements)
    graph.add(GPSPose2Factor(gpsModel, Symbol('x', 1), Point2(0, 0)));
    graph.add(GPSPose2Factor(gpsModel, Symbol('x', 2), Point2(5, 0)));
    graph.add(GPSPose2Factor(gpsModel, Symbol('x', 3), Point2(10, 0)));

    // initial varible values for the optimization
    // add random noise from ground truth values
    Values initials;
    initials.insert(Symbol('x', 1), Pose2(0.2, -0.3, 0.2));
    initials.insert(Symbol('x', 2), Pose2(5.1, 0.3, -0.1));
    initials.insert(Symbol('x', 3), Pose2(9.9, -0.1, -0.2));

    // print initial values
    initials.print("\nInitial Values:\n");

    // Use Gauss-Newton method optimizes the initial values
    GaussNewtonParams parameters;

    // print per iteration
    parameters.setVerbosity("ERROR");

    // optimize!
    GaussNewtonOptimizer optimizer(graph, initials, parameters);
    Values results = optimizer.optimize();

    // print final values
    results.print("Final Result:\n");


    // Calculate marginal covariances for all poses
    Marginals marginals(graph, results);

    // print marginal covariances
    cout << "x1 covariance:\n" << marginals.marginalCovariance(Symbol('x', 1)) << endl;
    cout << "x2 covariance:\n" << marginals.marginalCovariance(Symbol('x', 2)) << endl;
    cout << "x3 covariance:\n" << marginals.marginalCovariance(Symbol('x', 3)) << endl;
}
