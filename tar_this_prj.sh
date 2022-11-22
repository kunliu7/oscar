#!/bin/bash

# zip -r ../NIQAOAKit.zip ./*  -x  ./google_data/Google_Data/landscapes.pickl ./google_data/Google_Data/optimal-angles.pickl ./google_data/Google_Data/optimization.pickl ./data/*

tar -zcvf ../NIQAOAKit.tar.gz \
    --exclude="google_data/Google Data.zip" \
    --exclude=google_data/Google_Data/landscapes.pickl \
    --exclude=google_data/Google_Data/optimal-angles.pickl \
    --exclude=google_data/Google_Data/optimization.pickl \
    --exclude=data \
    ./
    