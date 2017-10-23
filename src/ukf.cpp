#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  DONE:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  Xsig_pred_.setZero();

  weights_.setZero(2 * n_aug_ + 1);
  weights_.setConstant(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  DONE:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if (!is_initialized_) {
    double px, py, v, psi, psi_dot;

    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      v = 0;
      psi = 0;
      psi_dot = 0;
    } else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      px = rho * cos(phi);
      py = rho * sin(phi);

      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);

      v = sqrt(vx * vx + vy * vy);

      psi = 0;
      psi_dot = 0;

    } else {
      throw invalid_argument("Cannot disable both radar and lidar");
    }
    x_ << px, py, v, psi, psi_dot;
    P_.setIdentity();

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    fallback_x_ = x_;
    fallback_P_ = P_;
  } else {
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
    Prediction(delta_t);

    if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    } else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    }

    // if unstable
    if (!isfinite(x_.sum()) || !isfinite(P_.sum())) {
      is_initialized_ = false;
      x_ = fallback_x_;
      P_ = fallback_P_;
    } else {
      fallback_x_ = x_;
      fallback_P_ = P_;
    }

    time_us_ = meas_package.timestamp_;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  DONE:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  Xsig_pred_ = GenerateAugmentedSigmaPoints();
  Xsig_pred_ = PredictSigmaPoints(Xsig_pred_, delta_t);

  // Predict mean
  x_ = Xsig_pred_ * weights_;

  // Predict covariance
  double w1 = lambda_ / (lambda_ + n_aug_);
  double w2 = 1 / (2 * (lambda_ + n_aug_));
  MatrixXd mean_subtracted = Xsig_pred_.colwise() - x_;
  for (int i = 0; i < mean_subtracted.cols(); ++i) {
    mean_subtracted(3, i) = NormalizeAngle(mean_subtracted(3, i));
  }
  MatrixXd temp = mean_subtracted * w2;
  temp.col(0) = temp.col(0) / w2 * w1;
  P_ = temp * mean_subtracted.transpose();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  DONE:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  int n_z = 2;

  // Measurement prediction
  VectorXd z_pred = VectorXd::Zero(n_z);
  // Measurement covariance
  MatrixXd S_ = MatrixXd::Zero(n_z, n_z);
  // Measurement sigma points
  MatrixXd z_sig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  // Prediction
  z_sig = Xsig_pred_.topRows(n_z);
  z_pred = z_sig * weights_;

  MatrixXd diff = z_sig.colwise() - z_pred;
  MatrixXd weighted_diff = diff.array().rowwise() * weights_.transpose().array();

  S_ = weighted_diff * diff.transpose();

  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;
  S_ += R;

  // Compute T and K
  MatrixXd X_diff = Xsig_pred_.colwise() - x_;
  MatrixXd Z_diff = z_sig.colwise() - z_pred;

  MatrixXd weighted_X_diff = X_diff.array().rowwise() * weights_.transpose().array();
  MatrixXd T = weighted_X_diff * Z_diff.transpose();

  MatrixXd K = T * S_.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  // Update x_ and P_
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  DONE:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  int n_z = 3;

  // Measurement prediction
  VectorXd z_pred = VectorXd::Zero(n_z);
  // Measurement covariance
  MatrixXd S_ = MatrixXd::Zero(n_z, n_z);
  // Measurement sigma points
  MatrixXd z_sig = MatrixXd::Zero(n_z, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double psi = Xsig_pred_(3, i);

    double v1 = cos(psi) * v;
    double v2 = sin(psi) * v;

    // rho
    z_sig(0, i) = sqrt(px * px + py * py);
    // psi
    z_sig(1, i) = atan2(py, px);
    // rho_dot
    z_sig(2, i) = (px * v1 + py * v2) / z_sig(0, i);
  }

  z_pred = z_sig * weights_;

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = z_sig.col(i) - z_pred;
    z_diff(1) = NormalizeAngle(z_diff(1));  // fix angle
    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;
  S_ += R;

  // Compute T and K
  MatrixXd X_diff = Xsig_pred_.colwise() - x_;
  MatrixXd Z_diff = z_sig.colwise() - z_pred;
  for (auto i = 0; i < Z_diff.cols(); ++i) {
    // yaw
    Z_diff(1, i) = NormalizeAngle(Z_diff(1, i));
    // yaw
    X_diff(3, i) = NormalizeAngle(X_diff(3, i));
  }

  MatrixXd weighted_X_diff = X_diff.array().rowwise() * weights_.transpose().array();
  MatrixXd T = weighted_X_diff * Z_diff.transpose();

  MatrixXd K = T * S_.inverse();
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  // angle normalization
  z_diff(1) = NormalizeAngle(z_diff(1));

  // Update x_ and P_
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();
}

MatrixXd UKF::GenerateAugmentedSigmaPoints() {
  MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.setZero();

  VectorXd x_aug(n_aug_);
  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.setZero();

  x_aug.setZero();
  x_aug.head(n_x_) = x_;

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  int n_new = n_aug_ - n_x_;

  MatrixXd Q = MatrixXd::Zero(n_new, n_new);
  Q(0, 0) = std_a_ * std_a_;
  Q(1, 1) = std_yawdd_ * std_yawdd_;
  P_aug.bottomRightCorner(n_new, n_new) = Q;

  MatrixXd sqrt_P_aug = P_aug.llt().matrixL();

  MatrixXd sqrt_term = sqrt(lambda_ + n_aug_) * sqrt_P_aug;

  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + sqrt_term.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - sqrt_term.col(i);
  }

  return Xsig_aug;
}

MatrixXd UKF::PredictSigmaPoints(const MatrixXd &Xsig_aug, const double delta_t) {
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    auto sigma_point = Xsig_aug.col(i);
    double px = sigma_point(0);
    double py = sigma_point(1);
    double v = sigma_point(2);
    double psi = sigma_point(3);
    double psi_dot = sigma_point(4);
    double nu_a = sigma_point(5);
    double nu_psi_dot = sigma_point(6);

    VectorXd x(5);

    if (fabs(psi_dot) > 1e-2) {
      x(0) = px + v / psi_dot * (sin(psi + psi_dot * delta_t) - sin(psi)) + 0.5 * delta_t * delta_t * cos(psi) * nu_a;
      x(1) = py + v / psi_dot * (-cos(psi + psi_dot * delta_t) + cos(psi)) + 0.5 * delta_t * delta_t * sin(psi) * nu_a;
    } else {
      x(0) = px + v * (v * cos(psi) * delta_t) + 0.5 * delta_t * delta_t * cos(psi) * nu_a;
      x(1) = py + v * (v * sin(psi) * delta_t) + 0.5 * delta_t * delta_t * sin(psi) * nu_a;
    }

    x(2) = v + 0 + delta_t * nu_a;
    x(3) = psi + psi_dot * delta_t + 0.5 * delta_t * delta_t * nu_psi_dot;
    x(4) = psi_dot + 0 + delta_t * nu_psi_dot;

    Xsig_pred.col(i) = x;
  }

  return Xsig_pred;
}

double UKF::NormalizeAngle(double angle) {
  if (angle > -M_PI && angle < M_PI) {
    return angle;
  } else {
    return angle - M_2_PI * floor((angle + M_PI) / M_2_PI);
  }
}
