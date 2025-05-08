# ei-ci-merge

## Instructions:


### Building the docker image

1. clone the project using git
```bash
git clone https://github.com/metabolomics-us/ei-ci-merge
```

2. build the docker image
```bash
docker build ./ -t ei-ci-merge
```


### Execute the docker image for merging a single file

```bash
docker run -v /$(pwd)/example:/data -it ei-ci-merge:latest -ci /data/Alkanes_1ng_3_CI.cdf -ei /data/Alkanes_1ng_3_EI.cdf -o /data/Alkanes_1ng_3_CI.csv
```

please ensure that this is done in the current directory and that you prefix you files with /data.

If you files are not located in the example directory, please replace

```bash
/$(pwd)/example
```

with an absolute path of your choice.

