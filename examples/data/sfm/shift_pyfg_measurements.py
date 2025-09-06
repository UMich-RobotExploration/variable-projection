"""
This is a small fixer file because all of the measurements were one-indexed but
the variables were zero-indexed. This will read in all of the .pyfg files and
shift the measurements to be zero-indexed.
"""

from py_factor_graph.io.pyfg_text import read_from_pyfg_text, save_to_pyfg_text
from py_factor_graph.factor_graph import FactorGraphData
import py_factor_graph.modifiers as modifiers
from copy import deepcopy

def shift_measurements(fg: FactorGraphData) -> FactorGraphData:
    """Shift all of the measurements in the factor graph to be zero-indexed."""
    new_pyfg = deepcopy(fg)
    new_pose_landmark_measures = []
    for measure in new_pyfg.pose_landmark_measurements:
        new_measure = deepcopy(measure)
        # e.g., "A101" -> "A100"
        pose_char = new_measure.pose_name[0]
        pose_index = int(new_measure.pose_name[1:]) - 1
        new_measure.pose_name = f"{pose_char}{pose_index}"
        landmark_char = new_measure.landmark_name[0]
        landmark_index = int(new_measure.landmark_name[1:]) - 1
        new_measure.landmark_name = f"{landmark_char}{landmark_index}"
        new_pose_landmark_measures.append(new_measure)
    new_pyfg.pose_landmark_measurements = new_pose_landmark_measures
    return new_pyfg

def get_shifted_line(line: str) -> str:
    """Shift a single line of a .pyfg file to be zero-indexed.

    Three possible line types:

    Pose variable
    VERTEX_SE3:QUAT 0.000000 A0 2.530868 1.256169 3.212721 -0.007525 -0.431229 0.162759 0.887409

    Landmark variable
    VERTEX_XYZ L0 0.000000 0.000000 0.000000

    Measurement
    EDGE_SE3_XYZ 0.0 A2014 L22485 -0.500772 3.387573 73.384285 0.504819 0.0 0.0 0.504819 0.0 0.504819
    """
    # if pose, return the line unchanged
    if line.startswith("VERTEX_SE3:QUAT"):
        return line

    # if landmark, return the line unchanged
    if line.startswith("VERTEX_XYZ"):
        return line

    # if measurement, shift the indices
    if line.startswith("EDGE_SE3_XYZ"):
        parts = line.split()
        pose_name = parts[2]
        landmark_name = parts[3]

        # shift pose name
        pose_char = pose_name[0]
        pose_index = int(pose_name[1:]) - 1
        new_pose_name = f"{pose_char}{pose_index}"

        # shift landmark name
        landmark_char = landmark_name[0]
        landmark_index = int(landmark_name[1:]) - 1
        new_landmark_name = f"{landmark_char}{landmark_index}"

        # reconstruct the line
        new_line = " ".join([parts[0], parts[1], new_pose_name, new_landmark_name] + parts[4:])
        return new_line

    raise ValueError(f"Line does not start with a recognized prefix: {line}")

def process_file(input_path: str, output_path: str) -> None:
    """Process a single .pyfg file to shift measurements to be zero-indexed."""
    with open(input_path, "r") as infile:
        lines = infile.readlines()

    new_lines = [get_shifted_line(line.strip()) for line in lines]

    with open(output_path, "w") as outfile:
        for line in new_lines:
            outfile.write(line + "\n")

def write_cleaned_files(input_files: list[str], output_files: list[str]) -> None:
    """Read in each input file, shift the measurements, and write to the output file."""
    for input_file, output_file in zip(input_files, output_files):
        print(f"Reading from {input_file} and writing to {output_file}...")
        process_file(input_file, output_file)
        print(f"Finished writing cleaned file to {output_file}")

# get all of the .bak.pyfg files recursively listed under input_dir
def get_backup_files(directory: str) -> list[str]:
    backup_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bak.pyfg"):
                backup_files.append(os.path.relpath(os.path.join(root, file), directory))

    if not backup_files:
        raise ValueError(f"No .bak.pyfg files found in directory {args.input_dir}.")
    return backup_files



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean all of the SfM .pyfg files to have measurements that are zero-indexed.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing all of the .pyfg files.")
    args = parser.parse_args()

    # make sure that input dir exists and is a directory
    import os
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist or is not a directory.")

    backup_files = get_backup_files(args.input_dir)

    pyfg_files = [fg_name.replace(".bak.pyfg", ".pyfg") for fg_name in backup_files]

    # see if we can read in all of the .pyfg files
    for fg_name in pyfg_files:
        fg_path = os.path.join(args.input_dir, fg_name)
        try:
            fg = read_from_pyfg_text(fg_path)
            unconnected_vars = fg.unconnected_variable_names
            if unconnected_vars:
                print(f"Warning: The factor graph {fg_path} has unconnected variables: {unconnected_vars}")
        except Exception as e:
            raise ValueError(f"Could not read in .pyfg file {fg_path}: {e}")
