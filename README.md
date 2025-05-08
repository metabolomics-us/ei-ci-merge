# ei-ci-merge

## Instructions:


### Docker:

#### Building the docker image

1. clone the project using git
```bash
git clone https://github.com/metabolomics-us/ei-ci-merge
```

2. build the docker image
```bash
docker build ./ -t ei-ci-merge
```


#### Execute the docker image for merging a single file

```bash
docker run -v /$(pwd)/example:/data -it ei-ci-merge:latest -ci /data/Alkanes_1ng_3_CI.cdf -ei /data/Alkanes_1ng_3_EI.cdf -o /data/Alkanes_1ng_3_CI.csv
```

please ensure that this is done in the current directory and that you prefix you files with /data.

If you files are not located in the example directory, please replace

```bash
/$(pwd)/example
```

with an absolute path of your choice.

### VENV

1. ensure you have python installed
2. download the source code

```bash
git clone https://github.com/metabolomics-us/ei-ci-merge
```

3. create a venv
4. activate the venv

```bash
source venv/bin/activate
```

5. install the requirements

```bash
pip install -r requirements
```

5. execute the program for the desired files

```bash
python3 merge.py -ci example/Alkanes_1ng_3_CI.cdf -ei example/Alkanes_1ng_3_EI.cdf -o merge.csv
```