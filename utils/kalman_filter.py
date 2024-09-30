## Copyright (C) Peizhi Yan. 2024

import numpy as np
from filterpy.kalman import KalmanFilter

# Function to initialize the Kalman Filter with matrices
def initialize_kalman_matrix(m, n, measure_noise=1e-5, process_noise=1e-5):
    """
    matrix size: m x n
    """
    kf = KalmanFilter(dim_x=m*n, dim_z=m*n)  # State and measurement dimensions are m*n (flattened matrix)
    
    # State transition matrix (F), assuming identity for simplicity (you can modify based on your system)
    kf.F = np.eye(m*n)  # Transition matrix (identity matrix, meaning no change in state)

    # Measurement matrix (H)
    kf.H = np.eye(m*n)  # Measurement matrix (observing the full state)

    # Process noise covariance (Q)
    kf.Q = np.eye(m*n) * process_noise  # Process noise

    # Measurement noise covariance (R)
    kf.R = np.eye(m*n) * measure_noise  # Measurement noise (adjust based on your scenario)

    # Initial state covariance (P)
    kf.P = np.eye(m*n) * 100  # Initial uncertainty in state

    # Initial state (X), assuming zero matrix (you can initialize with a prior estimate if available)
    kf.x = np.zeros((m*n, 1))  # Initial state (flattened n x n matrix)

    return kf


# Function to update the Kalman filter with matrix measurements in real-time
def kalman_filter_update_matrix(kf, measurement_matrix):
    m = measurement_matrix.shape[0]
    n = measurement_matrix.shape[1]

    # Check for invalid values in the measurement matrix
    if np.any(np.isnan(measurement_matrix)) or np.any(np.isinf(measurement_matrix)):
        print("Invalid measurement, skipping update")
        return None  # Skip update on invalid data

    # Flatten the matrix to fit into the Kalman filter
    measurement = measurement_matrix.flatten().reshape(-1, 1)  # Flatten and reshape to column vector

    # Predict the next state based on the previous state
    kf.predict()

    # Update with the new measurement (matrix)
    kf.update(measurement)

    # Extract the current estimated state (as a flattened matrix)
    estimated_state_flat = kf.x  # This is a flattened version of the n x n state matrix

    # Reshape back into an n x n matrix
    estimated_state_matrix = estimated_state_flat.reshape((m, n))

    return estimated_state_matrix



def main():
    # test code

    # Example of using the Kalman Filter in real-time with matrix measurements
    m = 1
    n = 3
    kf = initialize_kalman_matrix(m, n)  # Initialize Kalman filter for matrices

    # Example sequence of noisy matrix measurements
    measurements = [
        np.array([[1.1, 1.4, 1.9]]),
        np.array([[1.2, 1.2, 1.1]]),
        np.array([[1.0, 1.4, 0.6]]),
        np.array([[1.1, 1.3, 0.2]]),
        np.array([[1.2, 1.1, -0.1]]),
    ]

    for measurement_matrix in measurements:
        estimated_matrix = kalman_filter_update_matrix(kf, measurement_matrix)
        print(f"Updated Estimated Matrix:\n{estimated_matrix}")    


    pass



if __name__ == "__main__":
    main()
