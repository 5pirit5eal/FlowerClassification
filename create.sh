#!/bin/sh

# check if env directory exists
if [ -d ".venv/flower" ]; then
    # if it does, ask, if it really should be deleted
    read -n1 -p "This action will remove the old flower detection environment. Are you sure? [y,n] " REMOVE_FLAG

    # if it should be deleted, do that, otherwise abort the process
    case $REMOVE_FLAG in
        y|Y) printf "\nRemoving old flower detection environment.\n"
             rm -r .venv/flower ;;
        *) printf "\nAborting.\n" ;;
    esac
fi

# only create the new environment if it does not exists
if [ ! -d ".venv/flower" ]; then
    # create env
    python -m venv .venv/flower

    # activate env
    source .venv/flower/Scripts/activate

    # upgrade pip
    printf "\nUpgrade pip\n"
    python.exe -m pip install --upgrade pip
    pip install --upgrade setuptools

    # install target repo
    printf "\nInstall flower detection\n"
    pip install -r requirements.txt

    printf "\nEnvironment installation was successful\n"
fi
