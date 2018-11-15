echo "start sub：	$1";
echo "end sub：	$2";
echo "class:	$3";
echo "use baseline:	$4";
for((i=$1;i<=$2;i++));
do
	if [ $i -lt 10 ];
	then
		a=s0
	else
		a=s
	fi
	b=$i 
	sub=${a}${b}
	echo "Processing: $sub"
	python cv.py "$sub" $3 $4
done 