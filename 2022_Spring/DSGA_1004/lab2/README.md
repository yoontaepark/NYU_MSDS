# DS-GA 1004 Big Data
## Lab 2: Hadoop

*Handout date*: 2022-02-16

*Submission deadline*: 2022-03-04, 23:59 EST

## 0. Requirements

Sections 1 and 2 of this assignment are designed to get you familiar with HPC and the workflow of running Hadoop jobs.

For full credit on this lab assignment, you will need to provide working solutions for sections 4, and 5.
You are provided with small example inputs for testing these programs, but we will run your programs on larger data for grading purposes.
Be sure to commit your final working implementations to git and push your changes before the submission deadline!

## 1. High-performance Computing (HPC) at NYU

This lab assignment will require the use of the
Hadoop cluster run by the NYU high-performance
computing (HPC) center.  To learn more about HPC at
NYU, please refer to the [HPC Wiki](https://sites.google.com/nyu.edu/nyu-hpc).

By now, you should have received notification at
your NYU email address that your HPC account is active. If you have not received this notification yet, please contact the instructors immediately.

If you're new to HPC, please read through the
[tutorials](https://sites.google.com/nyu.edu/nyu-hpc/training-support/tutorials) section of the wiki, and - for this assignment in particular - the [MapReduce tutorial](https://sites.google.com/nyu.edu/nyu-hpc/training-support/tutorials/big-data-tutorial-map-reduce) section.

Logging into Peel on Linux or Mac from the NYU network is simple:
```bash
ssh netid@peel.hpc.nyu.edu
```
Uploading a file to Peel via SCP:
```bash
scp local_dir netid@peel.hpc.nyu.edu:peel_dir
```
Downloading a file from Peel via SCP:
```bash
scp netid@peel.hpc.nyu.edu:peel_dir local_dir
```

While it is possible to transfer files directly to and from Peel via SCP, we strongly recommend that you use git (and GitHub) to synchronize your code instead. This way, you can be sure that your submitted project is always up to date with the code being run on the HPC. To do this, you may need to set up a new SSH key (on Peel) and add it to your GitHub account; instructions for this can be found [here](https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account).

**Note**: Logging into the HPC from outside the NYU
network can be somewhat complicated.  Instructions
are given
[here](https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc).

## 2. Hadoop and `mrjob`

In lecture, we discussed the Map-Reduce paradigm in the abstract, and did not dive into the details of the Hadoop implementation.  Hadoop is an open-source implementation of map-reduce written in Java.
In this lab, you will be implementing map-reduce jobs using `mrjob`, a Hadoop wrapper library in Python.

### Environment setup

To setup the required environment, you need to execute the following command in the Git repo when you log into Peel:
```bash
source shell_setup.sh
```

These modifications add shortcuts for interacting with the Hadoop distributed filesystem (`hfs`) and launching map-reduce jobs (`hjs`), as well as set up useful environment variables for `mrjob`.

*Note*: For convenience, you can copy-paste the contents of that `shell_setup.sh` (or the `source` command itself, pointing to the correct path) into your `.bashrc` so that you don't need to re-run setup everytime you log in.

### Git on Peel

Follow [these instructions](https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh) to clone your Github repo through SSH. Cloning via HTTPS may cause problems with HPC and will be deprecated soon. The repository includes three problems on MapReduce:

## 3. A first map-reduce project (Not graded)

Included within the repository under `word_count/` is a full implementation of the "word-counting" program, and an example input text file (`book.txt`).  This implementation uses the `mrjob` Python package, which lets us write the map and reduce functions in python, and then handles all interaction with Hadoop for us.

The program consists of two files:
```
src/mr_wordcount.py
src/run_mrjob.sh
```

The shell script `run_mrjob.sh` loads the required module, `python/gcc/3.7.9`, and executes the MapReduceJob via Hadoop.
The python script accepts a few parameters, which are necessary to run correctly on Hadoop:

- `../book.txt` is the input file.
- `-r hadoop` indicates to `mrjob` that we are using a Hadoop backend. We can also simulate a Hadoop environment locally by removing this argument.
- `--hadoop-streaming-jar` tells `mrjob` where the Hadoop streaming file is located so that it can call the mappers and reducers appropriately.
- `--output-dir` indicates that the output of the MapReduce job should be placed in a specific folder (which you name) in HDFS. Keep in mind that duplicate filenames are not allowed on HDFS, so if you're using an existing folder name you'll have to remove the existing folder first.
- `--python-bin` and its value tell `mrjob` where the Python binary is so that the right version of Python can be used. This is the version loaded by the `module load` command.

The latest job result is then copied to the local file system and erased from the HDFS.



### Testing your mapper and reducer implementations without Hadoop

Before we move on, it's a good idea to run these programs directly so we know what to expect.  (*Hint*: this is also an easy way to debug, as long as you have a small input on hand!) Thankfully, `mrjob` makes our life easy: all we have to do is run the Python file containing our MapReduce job definition, and it will simulate the map-reduce workflow directly without running on the cluster.
You can run this by the following command:

```bash
python mr_wordcount.py ../book.txt
```

For simplicity, we have also included a shell script `run_mrjob_locally.sh` which you can execute directly.

After running this command, you should see the total counts of each word in `book.txt`.
Remember, we did this all on one machine without using Hadoop, but you should now have a sense of what a map-reduce job looks like.

### Launching word-count on Hadoop cluster

Now that we know how to run a map-reduce program locally, let's see how to run it on the cluster.
This is done by the other shell script, `run_mrjob.sh`, which as stated above, supplies the configuration variables necessary to instruct mrjob to run on HPC's Hadoop cluster.
When you run this script, you will see on the conole how the job is queued and run on the cluster, and you may expect this to take a bit longer to run than when executing locally.

After the job finishes, the result is stored in HDFS, which you can see by running:

```bash
hfs -ls word_count
```

You should see a few nested directories showing your job's results in several file "parts", each corresponding to a single reducer node.

To retrieve the results of the computation, run

```bash
hfs -get word_count
```

to get all the partial outputs, or if you want the entire output as one file, run

```bash
hfs -getmerge word_count word_count_total.out
```
After running these commands, the results of the computation will be available to you through the usual Unix file system.
The contents of `word_count_total.out` should match the output of your program when run locally, though the order of results may be different.

Once you have retrieved the results from HDFS, you can safely remove the old output:
```bash
hfs -rm -r word_count
```

At this point, you should now have successfully run your first Hadoop job!

## 4. Select, filter and aggregate

For your first MapReduce program, you will translate an SQL query into a map-reduce program.
In the `filter/` directory, you'll find some skeleton code in the `src` folder and some input data for your job.
The `movies.csv` file has one movie-genre combination per line with the format

```
movie name, movie genre
```

where if a movie belongs to several genres, there will be one line for each genre.
Your task is to count the number of movies in each genre, ignoring any lines that have the genre `Horror`, and retaining only the genres with 100 or more movies.
The SQL equivalent would be the following:

```sql
SELECT genre, count(distinct name) as num_movies
FROM movies
WHERE genre != 'Horror'
GROUP BY genre
HAVING num_movies >= 100
```

Your solution should be implemented by modifying the starter code.
We will run your solution on a supplemental dataset with different data of the same form as `movies.csv`.
Don't forget to commit your solution and push back to github!

## 5. Document similarity

In the last part of this assignment, you will develop a multi-stage map-reduce algorithm to compute similarity between all pairs of documents in a collection of text files.
The notion of "similarity" between documents that we will use is the *bag intersection*.
We denote by `A[w]` and `B[w]` the number of occurrences of word `w` in documents `A` and `B` respectively.
The bag intersection between `A` and `B` is computed by summing over all words the smaller of `A[w]` or `B[w]`:

```
sim(A, B) = sum_w min(A[w], B[w])
```

In the `docsim/` folder, you will find two subfolders, containing 1) a small collection of documents `docsim/data`, and 2) starter code `docsim/src`.
The starter code is based on the word counting example.
As illustrated in the shell script `docsim/src/run_mrjob.sh`, multiple separate input files can be provided on the command-line.
The starter code provides some code to identify the name of the current file being processed (by the mapper), and which should be used as the identifier for the document.

Unlike the previous examples, this starter code uses the [multi-step](https://mrjob.readthedocs.io/en/latest/guides/writing-mrjobs.html#multi-step-jobs) functionality of MRJob to automatically connect several stages of map-reduce processing in sequence.
We start with only a single step defined, but you are encouraged to add subsequent steps by including more `MRStep(...)` objects.
Please refer to the MRJob documentation for details on how to do this.

The final output of your program should be of the form `docID1, docID2, <similarity>` where `<similarity>` is as defined above.
The exact formatting of the output isn't important, but you should have one similarity score for each pair of documents.

Finally, include a brief description of your solution in the file `docsim/README.md`.
Your writeup should describe the inputs and outputs of each stage (including mappers and reducers for each step).
What problems, if any, did you encounter in designing your solution?
How well would your solution scale in terms of the number of documents?

### Tips

- Start small and work in incrementally: make sure that you can first count words in each document separately before moving on.
- A tiny dataset is included in `docsim/tiny`.  Use this for development.  It is small enough that you can compute the exact solution manually for reference.
- Use as few or as many map-reduce steps as you need.  Not every step needs to have both a mapper and a reducer.
- Think carefully about intermediate keys.  Use tuples if you need to.
- Make sure that your solution covers the situation where two documents have no words in common.
- Make sure that your solution produces `sim(A, A)` and that it correctly counts the number of words in the document.
- Make sure that your solution produces both `sim(A, B)` and `sim(B, A)`.  (And check that these results agree!)
