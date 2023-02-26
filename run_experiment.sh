if [ ! $# -eq 2 ]
then
    echo 'wrong param: 2 param in total (experiment python script, config file)'
    exit
fi

# run a python script with the corresponding config file
python $1 --config_file $2
