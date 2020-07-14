## Generate the schedule and module
```
python matmul.py -s
or
python conv2d.py -s
```

## Run the benchmark
```
perf stat -d --cpu=0-27 taskset -c 0-27 python matmul.py
or
perf stat -d --cpu=0-27 taskset -c 0-27 python conv2d.py

```
