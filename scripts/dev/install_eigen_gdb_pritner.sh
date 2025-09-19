project_dir=$(dirname -- "$0")/../../
project_dir="$(cd "${project_dir}" && pwd -P)"
printer_dir=${project_dir}/native_modules/subsampling/third-party/eigen_gdb_printer

# only copy if not already present
if [ ! -d "${printer_dir}" ]; then
    echo Copying Eigen GDB printer to ${printer_dir}

    mkdir -p ${printer_dir}
    cp -r ${project_dir}/native_modules/subsampling/third-party/eigen/debug/gdb/printers.py ${printer_dir}
    touch ${printer_dir}/__init__.py
fi

if ! grep -qs "register_eigen_printers" ~/.gdbinit; then
    echo Installing Eigen GDB printer to ~/.gdbinit

    echo "python" >> ~/.gdbinit
    echo "import sys" >> ~/.gdbinit
    echo "sys.path.insert(0, '${printer_dir}')" >> ~/.gdbinit
    echo "from printers import register_eigen_printers" >> ~/.gdbinit
    echo "register_eigen_printers(None)" >> ~/.gdbinit
    echo "end" >> ~/.gdbinit
fi
