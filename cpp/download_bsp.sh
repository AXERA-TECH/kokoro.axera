#!/bin/bash
if [ ! -d ax650n_bsp_sdk ]; then
  echo "clone ax650 bsp to ax650n_bsp_sdk, please wait..."
  git clone https://github.com/AXERA-TECH/ax650n_bsp_sdk.git
fi