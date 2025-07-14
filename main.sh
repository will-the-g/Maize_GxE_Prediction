#!/bin/sh

set -e

DIR=$(realpath $(dirname $0))

export PYTHONPATH=`pegasus-config --python`:`pegasus-config --python-externals`

#### Executing Workflow Generator ####
${DIR}/workflow.py -e condorpool -o $(pwd)/workflow.yml

# cat >> pegasus.properties <<EOF
# env.JAVA_HOME = /opt/homebrew/opt/openjdk
# pegasus.mode = development
# EOF

pegasus-plan --conf pegasus.properties \
    --dir submit \
    --sites condorpool \
    --output-sites local \
    --cleanup leaf \
    --force \
    "$@" workflow.yml

# Replace docker_init with container_init
# find submit/mayani/pegasus/maize-gxe/run0001 -name '*.sh' -exec sed -E -i '' -e "s@^docker_init (.*)@container_init ; cont_image='willtheg/maize:v1.2.4'@g" {} \;
