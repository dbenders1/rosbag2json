import json
import rosbag
import time
import yaml
import numpy as np
from pathlib import Path

FLOAT_TOL = 1e-6


def check_duplicated_timestamps(topic_name, timestamps):
    # Convert the list to a numpy array
    timestamps = np.array(timestamps)

    # Find unique timestamps and their first occurrence indices
    unique, indices = np.unique(timestamps, return_index=True)

    # Initialize a dictionary to store duplicated timestamps and their indices
    duplicates = {}

    # Iterate over unique timestamps
    for i, timestamp in enumerate(unique):
        # Find all occurrences of the current timestamp
        occurrence_indices = np.where(abs(timestamps - timestamp) < FLOAT_TOL)[0]

        # If there are more than one occurrence, it's a duplicate
        if len(occurrence_indices) > 1:
            duplicates[timestamp] = occurrence_indices

    # Print the duplicated timestamps and their indices
    if duplicates:
        print(f"Duplicate timestamps found in {topic_name}:")
    for timestamp, indices in duplicates.items():
        print(f"{timestamp} at indices {', '.join(map(str, indices))}")
    # if duplicates:
    #     exit(1)


def eta_to_dict(bag):
    topic_name = "/eta"

    # Determine size of eta
    for _, msg, _ in bag.read_messages(topic_name):
        n_eta = len(msg.y)
        break

    # Read messages from the bag
    # Create empty arrays if no messages are found
    n_msgs = bag.get_message_count(topic_name)
    if n_msgs == 0:
        t = np.array([])
        eta = np.array([])
    else:
        t = np.empty(n_msgs)
        eta = np.empty((n_eta, n_msgs))

        i = 0
        for _, msg, _ in bag.read_messages(topic_name):
            t[i] = msg.header.stamp.to_sec()
            eta[:, i] = np.array(msg.y[:n_eta])
            i = i + 1

        # Check for duplicated timestamps
        check_duplicated_timestamps(topic_name, t)

    # Create dictionary
    eta_dict = {}
    eta_dict["t"] = t.tolist()
    eta_dict["eta"] = eta.tolist()
    return eta_dict


def odometry_to_dict(bag):
    topic_name = "/falcon/ground_truth/odometry"

    # Read messages from the bag
    n_msgs = bag.get_message_count(topic_name)
    t = np.empty(n_msgs)
    p = np.empty((3, n_msgs))
    q = np.empty((4, n_msgs))
    v = np.empty((3, n_msgs))
    wb = np.empty((3, n_msgs))

    i = 0
    for _, msg, _ in bag.read_messages(topic_name):
        t[i] = msg.header.stamp.to_sec()
        p[:, i] = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        q[:, i] = np.array(
            [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ]
        )
        v[:, i] = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ]
        )
        wb[:, i] = np.array(
            [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ]
        )
        i = i + 1

    # Check for duplicated timestamps
    check_duplicated_timestamps(topic_name, t)

    # Create dictionary
    odom_dict = {}
    odom_dict["t"] = t.tolist()
    odom_dict["p"] = p.tolist()
    odom_dict["q"] = q.tolist()
    odom_dict["v"] = v.tolist()
    odom_dict["wb"] = wb.tolist()
    return odom_dict


def step_control_to_dict(bag):
    topic_name = "/step_control"

    # Read messages from the bag
    n_msgs = bag.get_message_count(topic_name)
    t = np.empty(n_msgs)
    u = np.empty((4, n_msgs))

    i = 0
    for _, msg, _ in bag.read_messages(topic_name):
        t[i] = msg.header.stamp.to_sec()
        u[:, i] = np.array(
            [
                msg.angular_velocities[0],
                msg.angular_velocities[1],
                msg.angular_velocities[2],
                msg.angular_velocities[3],
            ]
        )
        i = i + 1

    # Check for duplicated timestamps
    check_duplicated_timestamps(topic_name, t)

    # Create dictionary
    step_control_dict = {}
    step_control_dict["t"] = t.tolist()
    step_control_dict["u"] = u.tolist()
    return step_control_dict


def w_to_dict(bag):
    topic_name = "/w"

    # Determine size of w
    for _, msg, _ in bag.read_messages(topic_name):
        n_w = len(msg.y)
        break

    # Read messages from the bag
    # Create empty arrays if no messages are found
    n_msgs = bag.get_message_count(topic_name)
    if n_msgs == 0:
        t = np.array([])
        w = np.array([])
    else:
        t = np.empty(n_msgs)
        w = np.empty((n_w, n_msgs))

        i = 0
        for _, msg, _ in bag.read_messages(topic_name):
            t[i] = msg.header.stamp.to_sec()
            w[:, i] = np.array(msg.y[:n_w])
            i = i + 1

        # Check for duplicated timestamps
        check_duplicated_timestamps(topic_name, t)

    # Create dictionary
    w_dict = {}
    w_dict["t"] = t.tolist()
    w_dict["w"] = w.tolist()
    return w_dict


if __name__ == "__main__":
    # Start timing
    start = time.time()

    # User settings
    package_dir = Path(__file__).parents[1]
    json_dir = f"{package_dir}/data/converted_bags"
    config_dir = f"{package_dir}/config"
    config_path = f"{config_dir}/scripts/rosbag2json.yaml"

    # Read configuration parameters
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    bag_dir = config["recorded_data"]["bag_dir"]
    bag_names = config["recorded_data"]["bag_names"]
    topic_names = config["recorded_data"]["topic_names"]
    supported_topic_names = config["supported_topic_names"]

    # Process each topic in each bag and dump to a json file per bag
    for bag_name in bag_names:
        # Obtain bag file
        bag_path = f"{bag_dir}/{bag_name}.bag"
        print(f"\nConverting {bag_name}.bag")
        bag = rosbag.Bag(bag_path)

        # Store data in a dictionary
        bag_dict = {}
        for topic_name in topic_names:
            print(f"Processing topic: {topic_name}", end="\r")
            if topic_name in supported_topic_names:
                if topic_name == "/eta":
                    bag_dict[topic_name] = eta_to_dict(bag)
                elif topic_name == "/falcon/ground_truth/odometry":
                    bag_dict[topic_name] = odometry_to_dict(bag)
                elif topic_name == "/step_control":
                    bag_dict[topic_name] = step_control_to_dict(bag)
                elif topic_name == "/w":
                    bag_dict[topic_name] = w_to_dict(bag)
                print(f"Processing of topic {topic_name} completed")
            else:
                print(f"Processing of topic {topic_name} not supported; skipping")
        bag.close()

        # Write to json file
        json_path = f"{json_dir}/{bag_name}.json"
        with open(json_path, "w") as json_file:
            json.dump(bag_dict, json_file, indent=4)
        print(f"Converting of {bag_name}.bag completed")

    # End timing
    end = time.time()
    print(f"\nFinished converting bags in {end - start:.2f} seconds")
