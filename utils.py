import math


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_arrival_times(
    angle_deg, speed_of_sound=300, inter_ear_distance=0.3, source_distance=1.0
):
    """
    Calculates the arrival times of sound to two ear points given the source position in ms.
    Front = 0, right = 90, back = 180, left = 270 or -90 degrees.

    Parameters:
        angle_rad (float): Angle to the sound source in radians (0 = in front, positive = to the right).
        speed_of_sound (float): Speed of sound in the medium (e.g., 343 m/s in air).
        inter_ear_distance (float): Distance between the two ears (meters).
        source_distance (float): Distance from the center point between the ears to the sound source (meters).

    Returns:
        tuple: (arrival_time_left, arrival_time_right)
    """

    angle_rad = math.radians(angle_deg)

    # Ear positions (assuming head center at (0,0), ears on y-axis)
    left_ear = (0, -inter_ear_distance / 2)
    right_ear = (0, inter_ear_distance / 2)

    # Source position in Cartesian coordinates
    source_x = source_distance * math.cos(angle_rad)
    source_y = source_distance * math.sin(angle_rad)
    source_pos = (source_x, source_y)

    # Distance from source to each ear
    dist_left = euclidean(source_pos, left_ear)
    dist_right = euclidean(source_pos, right_ear)

    # Arrival times in ms
    time_left = dist_left / speed_of_sound * 1000
    time_right = dist_right / speed_of_sound * 1000

    return (time_left, time_right)


if __name__ == "__main__":
    # test
    angle = 90
    arrival_times = calculate_arrival_times(angle)
    print(
        f"Arrival times: Left Ear = {arrival_times[0]:.6f}ms, Right Ear = {arrival_times[1]:.6f}ms"
    )
