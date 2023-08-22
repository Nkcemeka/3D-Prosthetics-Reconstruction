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

if [ "$#" -eq 0 ] || [ "$#" -gt 2 ]; then
  echo "Number of command line arguments should be one or two!"
  echo "See USAGE details below!"
  usage
  exit 1
fi

# Get arguments and run the necessary commands
img_path="$1" # path to image
flip_cnt="${2:-0}" # flip contour for CAD viewing (0 is False)

# Check if image exists
if [ ! -f "$img_path" ]; then
  echo 'Image file does not exist'
  exit 1
fi

# Check if inputted number is correct
if ! [ "$flip_cnt" -eq 0 ] && ! [ "$flip_cnt" -eq 1 ]; then
  echo "Wrong integer passed; Input 0 or 1"
  exit 1
fi

# Check if python or python3 is available
# NOTE: Both aliases might work on Windows but not 
# necessarily on Linux

if command -v python &>/dev/null; then
  PYTHON=python
elif command -v python3 &>/dev/null; then
  PYTHON=python3
else
  echo "Python not found! Install or add to PATH!"
  exit 1
fi

# Run the Prosths file
$PYTHON prosths.py "$img_path" "$flip_cnt" 

# Generate the contours also
$PYTHON .dxf_writer.py .contours.csv obj_contour

