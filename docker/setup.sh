#!/bin/bash

HOME=/home

echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

apt-get update; apt-get install --no-install-recommends -yq \
    git lsb-core curl wget xvfb \
    python-pip python-dev \
    dirmngr gnupg2 python-setuptools
    
curl -sL https://deb.nodesource.com/setup_10.x | bash -
apt-get install -yq --no-install-recommends nodejs && npm install -g npm

./install_ros.sh
./clone_build.sh $HOME

./install_gzweb.sh $HOME

pip install --no-cache-dir -r requirements.txt

rm -rf /var/lib/apt/lists/*
