#!/bin/bash

# check arg count
if [ $# -ne 2 ]; then
    echo "usage: $0 product_name gen_internals(y|n)" 1>&2
    exit 1
fi

# check if include/compute & src/compute exist
if [ ! -d "$PWD/include/compute" ] || [ ! -d "$PWD/src/compute" ]; then
    echo "could not find one of src/compute or include/compute dirs. Please make sure to run in project root." 1>&2
    exit 1
fi

# sanitize input
OPERATION_NAME="$1"
GEN_INTERNALS="$2"

GEN_INTERNALS=$(echo $GEN_INTERNALS | tr '[:upper:]' '[:lower:]')
OPERATION_NAME_LOWER=$(echo $OPERATION_NAME | tr '[:upper:]' '[:lower:]')
OPERATION_NAME_FIRST_UPPER="$(tr '[:lower:]' '[:upper:]' <<< ${OPERATION_NAME_LOWER:0:1})${OPERATION_NAME_LOWER:1}"
OPERATION_INCLUDE_DIR="$PWD/include/compute/$OPERATION_NAME_LOWER"
OPERATION_SRC_DIR="$PWD/src/compute/$OPERATION_NAME_LOWER"

if [ "${GEN_INTERNALS:0:1}" != "y" ] && [ "${GEN_INTERNALS:0:1}" != "n" ]; then
    echo "invalid generate_internals argument. Please give y|n|Y|N|Yes|No|YES|NO." 1>&2
    exit 1
fi

# check if operation already exists
if [ -d "$OPERATION_INCLUDE_DIR" ] || [ -d "$OPERATION_SRC_DIR" ]; then
    echo "Operation already exists." 1>&2
    exit 1
fi

# get the proper 'in-place' flag for sed based on OS type
SED_INPLACE_FLAG=""
if [[ "$OSTYPE" == "linux-gnu " ]]; then
    SED_INPLACE_FLAG="-i"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SED_INPLACE_FLAG="-i ''"
else
    SED_INPLACE_FLAG="-i"
fi


printf "===== CREATING OPERATION \"%s\" =====\n" "${OPERATION_NAME_LOWER}"


# make the operation folder and copy templates
mkdir "$OPERATION_INCLUDE_DIR"
mkdir "$OPERATION_SRC_DIR"

echo "Creating $OPERATION_INCLUDE_DIR/${OPERATION_NAME_LOWER}op.h ..."
cp "$PWD/scripts/operation_templates/op_header_template.h" "$OPERATION_INCLUDE_DIR/${OPERATION_NAME_LOWER}op.h"

echo "Creating $OPERATION_SRC_DIR/${OPERATION_NAME_LOWER}op.cpp ..."
cp "$PWD/scripts/operation_templates/op_source_template.cpp" "$OPERATION_SRC_DIR/${OPERATION_NAME_LOWER}op.cpp"

if [ ${GEN_INTERNALS:0:1} = "y" ]; then
    echo "Creating $OPERATION_INCLUDE_DIR/${OPERATION_NAME_LOWER}_internal.h ..."
    cp "$PWD/scripts/operation_templates/op_internal_header_template.h" "$OPERATION_INCLUDE_DIR/${OPERATION_NAME_LOWER}_internal.h"

    echo "Creating $OPERATION_SRC_DIR/${OPERATION_NAME_LOWER}_internal.cpp ..."
    cp "$PWD/scripts/operation_templates/op_internal_source_template.cpp" "$OPERATION_SRC_DIR/${OPERATION_NAME_LOWER}_internal.cpp"

    echo "Creating $OPERATION_SRC_DIR/${OPERATION_NAME_LOWER}_internal_device.cu ..."
    cp "$PWD/scripts/operation_templates/op_internal_device_source_template.cu" "$OPERATION_SRC_DIR/${OPERATION_NAME_LOWER}_internal_device.cu"
fi

# replace template params with product name info
for file in $(ls $OPERATION_INCLUDE_DIR); do 
    echo "Generating source for $OPERATION_INCLUDE_DIR/$file ..."
    sed $SED_INPLACE_FLAG -e "s/<#OPERATION_NAME_LOWER#>/${OPERATION_NAME_LOWER}/g" -e "s/<#OPERATION_NAME#>/${OPERATION_NAME}/g" -e "s/<#OPERATION_NAME_FIRST_UPPER#>/${OPERATION_NAME_FIRST_UPPER}/g" "$OPERATION_INCLUDE_DIR/$file"
done
for file in $(ls $OPERATION_SRC_DIR); do 
    echo "Generating source for $OPERATION_SRC_DIR/$file ..."
    sed $SED_INPLACE_FLAG -e "s/<#OPERATION_NAME_LOWER#>/${OPERATION_NAME_LOWER}/g" -e "s/<#OPERATION_NAME#>/${OPERATION_NAME}/g" -e "s/<#OPERATION_NAME_FIRST_UPPER#>/${OPERATION_NAME_FIRST_UPPER}/g" "$OPERATION_SRC_DIR/$file"
done


# include header in compute/tensor_operations.h
echo "Including headers in library ..."
printf "\n#include \"%s/%sop.h\"" "${OPERATION_NAME_LOWER}" "${OPERATION_NAME_LOWER}" >> "$PWD/include/compute/tensor_operations.h"

# finished
printf "\nDONE"