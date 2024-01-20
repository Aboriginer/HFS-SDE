cd ~/.cache/torch_extensions/
first_file=$(ls | head -n 1)
ls $first_file/upfirdn2d
ls $first_file/fused/
rm -f $first_file/upfirdn2d/lock
rm -f $first_file/fused/lock