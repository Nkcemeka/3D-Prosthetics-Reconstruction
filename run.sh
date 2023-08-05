: '
Project Author: Chukwuemeka L. Nkama
Date: July 24, 2023

run.sh is a file that runs the prosths python
file by taking in certain cmd arguments
'

usage() {
  cat <<EOF
USAGE
------
./run.sh img_path (opt.arg: 0 or 1)
EOF
}

# print usage details to screen for user debugging
usage

# Get arguments and run the necessary commands
img_path="$1" # path to image
flip_cnt="${2:-0}" # flip contour for CAD viewing (0 is False)

# Run the Prosths file
python3 prosths.py "$img_path" "$flip_cnt" 

# Generate the contours also
python3 .dxf_writer.py .contours.csv obj_contour

