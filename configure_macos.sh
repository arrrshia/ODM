#!/bin/bash
uname=$(uname)
if [[ "$uname" != "Darwin" ]]; then
    echo "This script is meant for MacOS only."
    exit 1
fi

if [[ $2 =~ ^[0-9]+$ ]] ; then
    processes=$2
else
    processes=$(sysctl -n hw.ncpu)
fi

ensure_prereqs() {
    export DEBIAN_FRONTEND=noninteractive

    if ! command -v xcodebuild &> /dev/null; then
        echo "You need to install Xcode first. Go to the App Store and download Xcode"
        exit 1
    fi

    if ! command -v brew &> /dev/null; then
        echo "You need to install Homebrew first. https://brew.sh/"
        exit 1
    fi

}

installreqs() {
    ensure_prereqs
    
    brew install cmake gcc@12 python@3.8 tbb@2020 eigen gdal boost cgal libomp
    brew link tbb@2020

    python3.8 -m pip install virtualenv

    if [ ! -e ${RUNPATH}/venv ]; then
        python3.8 -m virtualenv venv
    fi

    source venv/bin/activate
    pip install --ignore-installed -r requirements.txt
}
    
install() {
    installreqs
    
    echo "Compiling SuperBuild"
    cd ${RUNPATH}/SuperBuild
    mkdir -p build && cd build
    cmake .. && make -j$processes

    cd /tmp
    pip download GDAL==3.6.2
    tar -xpzf GDAL-3.6.2.tar.gz
    cd GDAL-3.6.2
    if [ -e /opt/homebrew/bin/gdal-config ]; then
        python setup.py build_ext --gdal-config /opt/homebrew/bin/gdal-config
    else
        python setup.py build_ext --gdal-config /usr/local/bin/gdal-config
    fi
    python setup.py build
    python setup.py install
    rm -fr /tmp/GDAL-3.6.2 /tmp/GDAL-3.6.2.tar.gz

    cd ${RUNPATH}

    echo "Configuration Finished"
}

uninstall() {
    echo "Removing SuperBuild and build directories"
    cd ${RUNPATH}/SuperBuild
    rm -rfv build src download install
    cd ../
    rm -rfv build
}

reinstall() {
    echo "Reinstalling ODM modules"
    uninstall
    install
}

clean() {
    rm -rf \
        ${RUNPATH}/SuperBuild/build \
        ${RUNPATH}/SuperBuild/download \
        ${RUNPATH}/SuperBuild/src

    # find in /code and delete static libraries and intermediate object files
    find ${RUNPATH} -type f -name "*.a" -delete -or -type f -name "*.o" -delete
}

usage() {
    echo "Usage:"
    echo "bash configure.sh <install|update|uninstall|installreqs|help> [nproc]"
    echo "Subcommands:"
    echo "  install"
    echo "    Installs all dependencies and modules for running OpenDroneMap"
    echo "  reinstall"
    echo "    Removes SuperBuild and build modules, then re-installs them. Note this does not update OpenDroneMap to the latest version. "
    echo "  uninstall"
    echo "    Removes SuperBuild and build modules. Does not uninstall dependencies"
    echo "  installreqs"
    echo "    Only installs the requirements (does not build SuperBuild)"
    echo "  clean"
    echo "    Cleans the SuperBuild directory by removing temporary files. "
    echo "  help"
    echo "    Displays this message"
    echo "[nproc] is an optional argument that can set the number of processes for the make -j tag. By default it uses $(nproc)"
}

if [[ $1 =~ ^(install|installruntimedepsonly|reinstall|uninstall|installreqs|clean)$ ]]; then
    RUNPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    "$1"
else
    echo "Invalid instructions." >&2
    usage
    exit 1
fi
