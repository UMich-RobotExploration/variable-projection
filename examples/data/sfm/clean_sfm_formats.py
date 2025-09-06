
"""
SE3 pose format: VERTEX_SE3:QUAT <timestamp> <variable name> <x> <y> <z> <qx> <qy> <qz> <qw>
3D landmark format: VERTEX_XYZ <timestamp> <variable name> <x> <y> <z>
VERTEX_SE3:QUAT 0.000000 A0 -0.184888 -0.467706 -0.564942 -0.647987 0.741728 0.096877 0.143416

Measurement format: EDGE_SE3_XYZ <timestamp> <pose variable name> <landmark variable name> <x> <y> <z> <information matrix (6 values, upper triangular)>
EDGE_SE3_XYZ 0.0 A50 L248 0.064741 0.027717 1.526722 1.000000 0.000000 0.000000 1.000000 0.000000 1.000000
"""

def is_pose_line(line: str) -> bool:
    """Check if a line represents a pose."""
    return line.strip().startswith("VERTEX_SE3:QUAT")

def is_landmark_line(line: str) -> bool:
    """Check if a line represents a landmark."""
    return line.strip().startswith("VERTEX_XYZ")

def is_measurement_line(line: str) -> bool:
    """Check if a line represents a measurement."""
    return line.strip().startswith("EDGE_SE3_XYZ")

def clean_measurement_line(line: str) -> str:
    """Many of the measurement lines are corrupted because the information
    matrix is meant to be isotropic (scaled identity).

    We want to return all of the same values up to and including the z value,

    For the information, we want to read the last value (the information in z)
    and then set the rest of the diagonal values to be the same, and the
    off-diagonal values to be zero.
    """
    parts = line.strip().split()
    if len(parts) < 10:
        raise ValueError(f"Measurement line has too few parts: {line}")

    measurement, timestamp, pose_var, landmark_var = parts[:4]
    dx, dy, dz = parts[4:7]
    info_values = parts[7:]
    if len(info_values) != 6:
        raise ValueError(f"Measurement line has incorrect number of information values: {line}")

    # Use the last value (information in z) for all diagonal entries
    info_zz = info_values[-1]
    info_zz_formatted = f"{float(info_zz):.6f}"

    measurement_line = f"{measurement} {timestamp} {pose_var} {landmark_var} "
    measurement_line += f"{dx} {dy} {dz} {info_zz_formatted} 0.0 0.0 {info_zz_formatted} 0.0 {info_zz_formatted}"
    return measurement_line


def get_clean_line(line: str) -> str:
    """Remove comments and trailing whitespace from a line."""
    # if it starts with #, return empty string
    if line.strip().startswith("#"):
        return ""

    if is_pose_line(line):
        return line.strip()

    if is_landmark_line(line):
        # we need to drop the timestamp
        parts = line.strip().split()
        if len(parts) not in (5, 6):
            raise ValueError(f"Landmark line has incorrect number of parts: {line}")
        landmark, timestamp, var_name, x, y, z = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        return f"{landmark} {var_name} {float(x):.6f} {float(y):.6f} {float(z):.6f}"

    if is_measurement_line(line):
        return clean_measurement_line(line)

    raise ValueError(f"Line does not match any known format: {line}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean all of the SfM .txt files to be in a consistent format.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing all of the .txt files.")
    args = parser.parse_args()

    # make sure that input dir exists and is a directory
    import os
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory {args.input_dir} does not exist or is not a directory.")

    # get all of the .txt files recursively listed under input_dir
    def get_txt_files(directory: str) -> list[str]:
        txt_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    txt_files.append(os.path.relpath(os.path.join(root, file), directory))
        return txt_files

    txt_files = get_txt_files(args.input_dir)
    if not txt_files:
        raise ValueError(f"No .txt files found in directory {args.input_dir}.")

    # all of the output files will be the same name but with .pyfg extension
    output_files = [f.replace(".txt", ".pyfg") for f in txt_files]

    for input_file, output_file in zip(txt_files, output_files):
        input_file = os.path.join(args.input_dir, input_file)
        output_file = os.path.join(args.input_dir, output_file)
        print(f"Cleaning {input_file} and writing to {output_file}")
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for line in infile:
                try:
                    clean_line = get_clean_line(line)
                    if clean_line:
                        outfile.write(clean_line + "\n")
                except ValueError as e:
                    print(f"Warning: {e}")

        print(f"Finished writing cleaned file to {output_file}")