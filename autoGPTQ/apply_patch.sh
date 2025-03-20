#!/bin/bash

install_dir=$(pip show vllm | grep "Location" | awk '{print $2}')"/"

if [ -z "$install_dir" ]; then
    echo "Unable to find the installation path for vlllm library. Please ensure that the library is installed correctly."
    exit 1
fi

echo "vllm install dir: $install_dir"

cp vllm-0.6.5-gptq.patch "$install_dir"
cd "$install_dir"
git apply vllm-0.6.5-gptq.patch

if [[ $? -eq 0 ]];then
    echo "success done."
else
    echo "error occur."
fi

rm -rf vllm-0.6.5-gptq.patch
