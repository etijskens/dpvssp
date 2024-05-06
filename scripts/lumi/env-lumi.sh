# This file must be sourced

module purge

# default modules
ml LUMI/23.09
ml partition/C
ml PrgEnv-cray/8.4.0
ml buildtools/23.09
ml cray-python/3.10.10
ml cray-libsci

for cla in "$@"
do
  if [[ $cla -eq -p ]]
  then
    # load modules for performance analysis
    ml perftools-base
    ml perftools-lite
  fi
done

ml