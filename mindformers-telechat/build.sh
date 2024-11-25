#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "---------------- MindFormers: build start ----------------"
BASEPATH=$(cd "$(dirname $0)"; pwd)

export BUILD_PATH="${BASEPATH}/build/"

python setup.py bdist_wheel -d ${BASEPATH}/output

if [ ! -d "${BASEPATH}/output" ]; then
    echo "The directory ${BASEPATH}/output dose not exist."
    exit 1
fi

cd ${BASEPATH}/output || exit
for package in mindformers*whl
do
    [[ -e "${package}" ]] || break
    sha256sum ${package} > ${package}.sha256
done
pip install mindformers*whl -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ${BASEPATH} || exit
rm -rf *-info
echo "---------------- MindFormers: build and install end   ----------------"
