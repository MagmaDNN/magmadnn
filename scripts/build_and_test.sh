#!/bin/bash

# This file will clean, build, and run the testers for the library

set -o errexit -o pipefail -o noclobber -o nounset

OPTIND=1
verbose=0
n_threads=1
debug=0
clean=0
do_test=1

function check_err() {
    if [ $? != 0 ]; then
        if [ $verbose = 1 ]; then
            printf "[Error]: Build failed on \"%s\".\n" "$1"
            exit $?
        fi
    fi
}

function show_help() {
    printf "Usage: %s ... options.\n" "$0"
    printf "\t-h|--help help; show available options.\n"
    printf "\t-j|--threads num n_threads; set the number of threads to use in building.\n"
    printf "\t-d|--debug debug; set whether to build in debug mode or not.\n"
    printf "\t-o|--output out output_file; specify the output file.\n"
    printf "\t-c|--clean clean; clean the library before building.\n"
    printf "\t-v|--verbose verbose; show output.\n"
    printf "\t--no-test run_tests; if set, then the testers won't be built or run.\n"
    printf "\t--dev dev; development mode, same as '--debug --clean --verbose'\n"
}

! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    printf "[Error]: enhanced getopt not available.\n"
    exit 1
fi

OPTIONS=hj:dco:v
LONGOPTIONS=help,threads:,debug,clean,output:,verbose,no-test,dev

! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    exit 2
fi

eval set -- "$PARSED"

while true; do
    case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    -j|--threads)
        n_threads="$2"
        shift 2
        ;;
    -d|--debug)
        debug=1
        shift
        ;;
    -c|--clean)
        clean=1
        shift
        ;;
    -o|--output)
        OUT=">$2"
        shift 2
        ;;
    -v|--verbose)
        verbose=1
        shift
        ;;
    --no-test)
        do_test=0
        shift
        ;;
    --dev)
        debug=1
        clean=1
        verbose=1
        shift
        ;;
    --)
        shift
        break
        ;;
    *)
        printf "[Error]: Unknown option.\n"
        show_help
        exit 3
        ;;
    esac
done


shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

# clean the library
if [ $clean = 1 ]; then
    if [ $verbose = 1 ]; then
        make clean 
    else
        make clean >/dev/null
    fi
    check_err "make clean"
fi

# build the library
if [ $verbose = 1 ]; then
    make -j $n_threads DEBUG=$debug
else
    make -j $n_threads DEBUG=$debug >/dev/null
fi
check_err "make"

# install the library
if [ $verbose = 1 ]; then
    make install DEBUG=$debug
else
    make install DEBUG=$debug >/dev/null
fi
check_err "make install"


if [ $do_test = 1 ]; then
    # build the testers
    if [ $verbose = 1 ]; then
        make testing -j $n_threads DEBUG=$debug
    else
        make testing -j $n_threads DEBUG=$debug >/dev/null
    fi
    check_err "make testing"

    # run testers
    TESTING_FILES=$(cd testing/bin && ls)
    for i in $TESTING_FILES; do
        if [ $verbose = 1 ]; then
            printf "=====TESTING \"%s\"=====\n" "$i"
            ./testing/bin/$i 
            check_err "tester $i"
            printf "\n"
        else
            ./testing/bin/$i >/dev/null
        fi
    done
fi


# success
if [ $verbose = 1 ]; then
    printf "Success!\n"
fi
